using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads
using DelimitedFiles  # for CSV
using Dates

model_path = joinpath(@__DIR__, "humanoid.xml")
model = MuJoCo.load_model(model_path)
data = MuJoCo.init_data(model)

# Constants for MPPI
const Position = [2.0, 0.0]
const K = 30
const T = 75
const λ = 1.0
const Σ = 0.75

const nx = length(data.qpos) + length(data.qvel)
const nu = length(data.ctrl)

# === Logging Setup ===

save_timestamp = Dates.format(now(), "yyyy-mm-dd_HHMMSS")
const SAVE_DIR = joinpath("data", save_timestamp)
isdir(SAVE_DIR) || mkpath(SAVE_DIR)

const LOG_STATES = Vector{Vector{Float64}}()
const LOG_ACTIONS = Vector{Vector{Float64}}()
const LOG_TIMES = Float64[]

function log_data!(d::MuJoCo.Data, u::Vector{Float64})
    push!(LOG_TIMES, d.time)
    push!(LOG_STATES, vec(vcat(d.qpos, d.qvel)))  # full state
    push!(LOG_ACTIONS, copy(u))              # control
end

function humanoid_cost(qpos, qvel, ctrl, t)
    cost = 0.0

    # Extract torso position and orientation
    root_pos = qpos[1:3]  # x, y, z position
    target_pos = Position # position of the target

    torso_quat = qpos[4:7] # quaternion orientation

    # Desired forward velocity (x-direction)
    root_lin_vel = qvel[1:2] # only xy
    target_vel = [0.4, 0.0]  # only xy

    # Weights (can tune these)
    # Penalize torso rotation (try to keep it upright)
    # Convert quaternion to roll and pitch
    roll = atan(2(torso_quat[1] * torso_quat[2] + torso_quat[3] * torso_quat[4]),
        1 - 2(torso_quat[2]^2 + torso_quat[3]^2))
    pitch = asin(2(torso_quat[1] * torso_quat[3] - torso_quat[4] * torso_quat[2]))
    cost += 5.0 * (roll^2 + pitch^2)  # Keep upright and restrict pitch rotation further

    # Restric Yaw rotation
    yaw = atan(2(torso_quat[1] * torso_quat[4] + torso_quat[2] * torso_quat[3]),
           1 - 2(torso_quat[3]^2 + torso_quat[4]^2))
    cost += 0.1 * yaw^2  # Penalize deviation from facing forward

    # Penalize distance from goal (xy)
    cost += 12.5 * norm(root_pos[1:2] - target_pos[1:2])

    # Penalize torso height
    target_height = 1.28 #1.282 = initial_qpos_root_z
    cost += 3.0 * (target_height - root_pos[3])  # softly keep torso up

    # Penalize torso velocity difference
    cost += 1.0 * norm(root_lin_vel - target_vel)

    # Get body IDs
    id_left = MuJoCo.body(model, "shin_left").id
    id_right = MuJoCo.body(model, "shin_right").id

    vx_left = get_body_vx(data, id_left)
    vx_right = get_body_vx(data, id_right)

    if vx_left > vx_right
        foot_swing = "foot_left"
        foot_stance = "foot_right"
        knee_swing = id_left
    else
        foot_swing = "foot_right"
        foot_stance = "foot_left"
        knee_swing = id_right
    end

    swing_id = MuJoCo.body(model, foot_swing).id
    stance_id = MuJoCo.body(model, foot_stance).id

    swing_foot = data.xpos[swing_id + 1, 1] # position of foot's current x
    foot_targetx = root_pos[1] + 0.5  # 30cm ahead of torso
    cost += 7.5 * (swing_foot - foot_targetx)^2

    swing_knee = data.xpos[knee_swing + 1, 1] # position of knee's current 3
    cost += 3.5 * (swing_knee - foot_targetx)^2

    swing_foot_z = data.xpos[swing_id + 1, 3] # position of swing foot's current z
    knee_targetz = root_pos[3] - 0.4  # below 40cm of torso
    #cost += 2.0 * (swing_knee - knee_targetz)^2
    
    stance_foot_z = data.xpos[stance_id + 1, 3] # position of stance foot's current z
    foot_clearance = swing_foot_z - stance_foot_z
    if foot_clearance < 0
        cost -= 0.01 * foot_clearance
    end

    left_foot_y = data.xpos[MuJoCo.body(model, "foot_left").id + 1, 2] # position of left foot's current y
    right_foot_y = data.xpos[MuJoCo.body(model, "foot_right").id + 1, 2] # position of right foot's current y
    leg_clearance = left_foot_y - right_foot_y
    if leg_clearance < 0
        cost -= 1.0 * leg_clearance
    end

    cost += 0.01 * sum(ctrl .^ 2)

    return cost
end

function terminal_cost(qpos, qvel, t)
    return 10.0 * humanoid_cost(qpos, qvel, zeros(nu), t)
end

const U_global = zeros(nu, T)
const d_copies = [init_data(model) for _ in 1:nthreads()]
const temp = zeros(nu)

function rollout(m::Model, d::Data, U::Matrix{Float64}, noise::Array{Float64,3})
    costs = zeros(K)
    @threads for k in 1:K
        d_copy = d_copies[threadid()]
        copyto!(d_copy.qpos, d.qpos)
        copyto!(d_copy.qvel, d.qvel)

        cost = 0.0
        for t in 1:T
            @views ctrl = d_copy.ctrl
            @views ctrl .= U[:, t]
            @views ctrl .+= noise[:, t, k]
            mj_step(m, d_copy)
            cost += humanoid_cost(d_copy.qpos, d_copy.qvel, d_copy.ctrl, t)
        end
        costs[k] = cost + terminal_cost(d_copy.qpos, d_copy.qvel, T)
    end
    return costs
end

function mppi_step!(m::Model, d::Data)
    noise = randn(nu, T, K) * Σ
    costs = rollout(m, d, U_global, noise)
    β = minimum(costs)
    weights = exp.(-1 / λ * (costs .- β))
    weights ./= sum(weights)

    for t in 1:T
        fill!(temp, 0.0)
        for k in 1:K
            @views temp .+= weights[k] .* noise[:, t, k]
        end
        @views U_global[:, t] .+= temp
    end
end

function mppi_controller!(m::Model, d::Data)
    mppi_step!(m, d)
    d.ctrl .= U_global[:, 1]

    # Log state and control
    log_data!(d, U_global[:, 1])

    # Shift control sequence
    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= 0.1 * U_global[:, end-1]
end

function save_logs()
    state_array = reduce(hcat, LOG_STATES)'  # T × nx
    action_array = reduce(hcat, LOG_ACTIONS)'  # T × nu
    time_array = collect(LOG_TIMES)

    writedlm(joinpath(SAVE_DIR, "states.csv"), state_array, ',')
    writedlm(joinpath(SAVE_DIR, "actions.csv"), action_array, ',')
    writedlm(joinpath(SAVE_DIR, "times.csv"), time_array, ',')

    println("Log data saved to: $SAVE_DIR")
end


# === Visualization and Save Hook ===
init_visualiser()
try
    visualise!(model, data; controller=mppi_controller!)
finally
    save_logs()
end


# Save logged data after simulation
atexit() do
    state_array = reduce(hcat, LOG_STATES)'  # T × nx
    action_array = reduce(hcat, LOG_ACTIONS)'  # T × nu
    time_array = collect(LOG_TIMES)

    writedlm(joinpath(SAVE_DIR, "states.csv"), state_array, ',')
    writedlm(joinpath(SAVE_DIR, "actions.csv"), action_array, ',')
    writedlm(joinpath(SAVE_DIR, "times.csv"), time_array, ',')

    println("Logged data saved to: $(SAVE_DIR)")
end
