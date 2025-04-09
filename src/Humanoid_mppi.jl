
using MuJoCo
using LinearAlgebra
using Random
using Statistics
using Base.Threads

model_path = joinpath(@__DIR__, "humanoid.xml")
model = MuJoCo.load_model(model_path)
data = MuJoCo.init_data(model)

#println("Fields in data struct:")
#println(fieldnames(typeof(data)))

#println("\nProperties available on data object:")
#println(propertynames(data))

#println("nq = ", length(data.qpos))
#println("nv = ", length(data.qvel))
#println("nu = ", length(data.ctrl))  # Usually same as model.nu

const K = 50   # num sample trajectories
const T = 100   # horizon
const λ = 1.0   # temperature
const Σ = 1.0   # control noise for exploration

const nx = length(data.qpos)+length(data.qvel)
const nu = length(data.ctrl)


function humanoid_cost(qpos, qvel, ctrl, t)
    cost = 0.0

    # Extract torso position and orientation
    root_pos = qpos[1:3]  # x, y, z position
    target_pos = [2.0, 0.0] # simple locomotion

    torso_quat = qpos[4:7] # quaternion orientation

    # Desired forward velocity (x-direction)
    root_lin_vel = qvel[1:2] # only xy
    target_vel = [0.5, 0.0]  # only xy

    # Weights (can tune these)
    # Penalize torso rotation (try to keep it upright)
    # Convert quaternion to roll and pitch
    roll = atan(2(torso_quat[1] * torso_quat[2] + torso_quat[3] * torso_quat[4]),
        1 - 2(torso_quat[2]^2 + torso_quat[3]^2))
    pitch = asin(2(torso_quat[1] * torso_quat[3] - torso_quat[4] * torso_quat[2]))
    cost += 5.0 * (roll^2 + pitch^2)  # Keep upright and restrict pitch rotation further

    # Penalize distance from goal (xy)
    cost += 12.0 * norm(root_pos[1:2] - target_pos[1:2])

    # Penalize torso height
    target_height = 1.28 #1.282 = initial_qpos_root_z
    cost += 2.25 * (target_height - root_pos[3])  # softly keep torso up

    # Penalize torso velocity difference
    cost += 1.0 * norm(root_lin_vel - target_vel)

    #foot_left_id = MuJoCo.body(model, "foot_left").id
    #foot_right_id = MuJoCo.body(model, "foot_right").id
    #ankle_left_xvel = qvel[21]
    #ankle_right_xvel = qvel[15]
    #if (ankle_left_xvel * ankle_right_xvel) < 0
    #    if ankle_left_xvel > 0
    #        # LEFT FOOT swings, RIGHT FOOT supports
    #        foot_swing = "foot_left"
    #        foot_stance = "foot_right"
    #    elseif ankle_right_xvel > 0
    #        # RIGHT FOOT swings, LEFT FOOT supports
    #        foot_swing = "foot_right"
    #        foot_stance = "foot_left"

    step_period = 100  # number of timesteps per full step cycle
    phase = t % step_period
    
    if phase < step_period ÷ 2
        # LEFT FOOT swings, RIGHT FOOT supports
        foot_swing = "foot_left"
        foot_stance = "foot_right"
    else
        # RIGHT FOOT swings, LEFT FOOT supports
        foot_swing = "foot_right"
        foot_stance = "foot_left"
    end

    swing_id = MuJoCo.body(model, foot_swing).id
    swing_pos = data.xpos[swing_id + 1, 1] # position of foot's current x
    foot_target = root_pos[1] + 0.5  # 50cm ahead of torso
    cost += 10.0 * (swing_pos - foot_target)^2

    stance_id = MuJoCo.body(model, foot_stance).id
    stance_pos_z = data.xpos[stance_id + 1, 3] # position of stance foot's current z
    swing_pos_z = data.xpos[swing_id + 1, 3] # position of swing foot's current z
    foot_gap = 0.01 # 20cm
    cost += 0.01 * (stance_pos_z - swing_pos_z)
    #if swing_pos_z- stance_pos_z < foot_gap
    #    cost += 7.5 * (foot_gap - (stance_pos_z - swing_pos_z))
    #end

    stance_pos_y = data.xpos[stance_id + 1, 2] # position of stance foot's current y
    swing_pos_y = data.xpos[swing_id + 1, 2] # position of swing foot's current y
    leg_gap = 0.4 # 20cm
    cost += 0.1 * norm(stance_pos_y - swing_pos_y)
    #if norm(stance_pos_y - swing_pos_y) < leg_gap
    #    cost += 8.5 * (leg_gap - norm(stance_pos_y - swing_pos_y))
    #end

    #stance_id = MuJoCo.body(model, foot_stance).id
    #jac = mj_zeros(3, model.nv)
    #mj_jacBodyCom(model, data, jac, nothing, stance_id)
    #stance_vel = jac * data.qvel
    #cost += 10.0 * stance_vel[3]^2  # penalize vertical motion

    # Penalize control effort
    cost += 0.01 * sum(ctrl .^ 2)

    return cost
end

function running_cost(x_pos, theta, x_vel, theta_vel, control)
    cart_pos_cost = 1.0 * x_pos^2
    pole_pos_cost = 20.0 * (cos(theta) - 1.0)^2  # Changed to use angle directly
    cart_vel_cost = 0.1 * x_vel^2
    pole_vel_cost = 0.1 * theta_vel^2
    ctrl_cost = 0.01 * control[1]^2
    return cart_pos_cost + pole_pos_cost + cart_vel_cost + pole_vel_cost + ctrl_cost
end


# makes it not kiss the corners
function terminal_cost(qpos, qvel, t)
    return 10.0 * humanoid_cost(qpos, qvel, zeros(nu), t)
end

# init controls
const U_global = zeros(nu, T)

function rollout(m::Model, d::Data, U::Matrix{Float64}, noise::Array{Float64,3})
    costs = zeros(K)
    # thanks claude san for making this multi thread?
    @threads for k in 1:K
        d_copy = init_data(m)
        d_copy.qpos .= d.qpos
        d_copy.qvel .= d.qvel
        cost = 0.0
        for t in 1:T
            # Apply control with noise
            d_copy.ctrl .= U[:, t] + noise[:, t, k]
            mj_step(m, d_copy)
            # Extract state information
            # Compute running cost
            cost += humanoid_cost(d_copy.qpos, d_copy.qvel, d_copy.ctrl, t)
        end
        # Add terminal cost
        costs[k] = cost + terminal_cost(
            d_copy.qpos, d_copy.qvel, T
        )
    end
    return costs
end

function mppi_step!(m::Model, d::Data)
    # Generate noise
    noise = randn(nu, T, K) * Σ

    costs = rollout(m, d, U_global, noise)
    β = minimum(costs)
    weights = exp.(-1 / λ * (costs .- β))
    weights ./= sum(weights)
    # update controls
    for t in 1:T
        U_global[:, t] .= U_global[:, t] + sum(weights[k] * noise[:, t, k] for k in 1:K)
    end
end

function mppi_controller!(m::Model, d::Data)
    mppi_step!(m, d)
    d.ctrl .= U_global[:, 1]
    # shifting controls
    U_global[:, 1:end-1] .= U_global[:, 2:end]
    U_global[:, end] .= 0.1 * U_global[:, end-1]  # Smaller decay factor
end

# woohoooo
init_visualiser()
visualise!(model, data; controller=mppi_controller!)
