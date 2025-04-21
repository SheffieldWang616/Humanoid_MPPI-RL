using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads
using DelimitedFiles  # for CSV
using Dates

model_path = joinpath(@__DIR__, "..", "models", "cartpole.xml")
model = MuJoCo.load_model(model_path)
data = MuJoCo.init_data(model)

# # Constants for MPPI
# const Position = Ref(1.0)
# const goal_step = 1.0  
# goal_counter = 0  # Counter for goal reached
# const goal_threshold = 0.1  # Distance threshold to detect "goal reached"

const K = 75  # num sample trajectories
const T = 100 # horizon
const λ = 1.0   # temperature
const Σ = 0.75  # control noise for exploration

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


function get_body_vx(data, body_id)
    start = body_id * 6 - 5  # body_id-th 6D block
    return data.cvel[start + 3]  # vx is the 4th value (index 3)
end

function running_cost(x_pos, theta, x_vel, theta_vel, control)
    cart_pos_cost = 1.0 * x_pos^2
    pole_pos_cost = 20.0 * (cos(theta) - 1.0)^2  # Changed to use angle directly
    cart_vel_cost = 0.1 * x_vel^2
    pole_vel_cost = 0.1 * theta_vel^2
    ctrl_cost = 0.01 * control[1]^2
    # goal_cost = 20.0 * (x_pos - Position[])^2
    return cart_pos_cost + pole_pos_cost + cart_vel_cost + pole_vel_cost + ctrl_cost
end


# makes it not kiss the corners
function terminal_cost(x_pos, theta, x_vel, theta_vel)
    return 10.0 * running_cost(x_pos, theta, x_vel, theta_vel, zeros(nu))
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

            x_pos = d_copy.qpos[1]
            theta = d_copy.qpos[2]
            x_vel = d_copy.qvel[1]
            theta_vel = d_copy.qvel[2]

            cost += running_cost(x_pos, theta, x_vel, theta_vel, d_copy.ctrl)
        end

        costs[k] = cost + terminal_cost(
            d_copy.qpos[1], d_copy.qpos[2], d_copy.qvel[1], d_copy.qvel[2]
        )
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

    # # Goal-switching logic based on cart x-position
    # x_pos = d.qpos[1]
    # if abs(x_pos - Position[]) < goal_threshold
    #     global goal_counter
    #     goal_counter += 1
    #     Position[] = (-1)^goal_counter * ceil(goal_counter / 2) * goal_step
    #     println("Goal Reached: ", goal_counter, " Times. New goal position: ", Position)
    # end

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
