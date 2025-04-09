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
    root_pos = qpos[1:3]
    target_pos = Position

    torso_quat = qpos[4:7]
    root_lin_vel = qvel[1:2]
    target_vel = [0.5, 0.0]

    roll = atan(2(torso_quat[1] * torso_quat[2] + torso_quat[3] * torso_quat[4]),
                1 - 2(torso_quat[2]^2 + torso_quat[3]^2))
    pitch = asin(2(torso_quat[1] * torso_quat[3] - torso_quat[4] * torso_quat[2]))
    cost += 5.0 * (roll^2 + pitch^2)
    cost += 12.0 * norm(root_pos[1:2] - target_pos[1:2])
    cost += 2.25 * (1.28 - root_pos[3])
    cost += 1.0 * norm(root_lin_vel - target_vel)

    step_period = 100
    phase = t % step_period
    foot_swing, foot_stance = phase < step_period ÷ 2 ? ("foot_left", "foot_right") : ("foot_right", "foot_left")

    swing_id = MuJoCo.body(model, foot_swing).id
    stance_id = MuJoCo.body(model, foot_stance).id
    swing_pos = data.xpos[swing_id + 1, 1]
    foot_target = root_pos[1] + 0.5
    cost += 10.0 * (swing_pos - foot_target)^2

    stance_pos_z = data.xpos[stance_id + 1, 3]
    swing_pos_z = data.xpos[swing_id + 1, 3]
    cost += 0.01 * (stance_pos_z - swing_pos_z)

    stance_pos_y = data.xpos[stance_id + 1, 2]
    swing_pos_y = data.xpos[swing_id + 1, 2]
    cost += 0.1 * norm(stance_pos_y - swing_pos_y)

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
