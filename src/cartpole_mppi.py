import mujoco
import numpy as np
import math
import os

# === Load model and data ===
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "cartpole.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# === Constants ===
K = 30     # num sample trajectories
T = 100    # horizon
_lambda = 1.0  # temperature
sigma = 1.0     # control noise

nx = 2 * model.nv
nu = model.nu

# === Cost Functions ===

def cost(qpos, qvel, ctrl):
    cost = 0.0

    # torso_pos = qpos[0:3]     # x, y, z
    torso_quat = qpos[3:7]      # quaternion orientation

    target_vel_x = 0.5
    current_vel_x = qvel[0]
    cost += 1.0 * (current_vel_x - target_vel_x)**2
    cost += 2.0 * qvel[1]**2

    # Convert quaternion to roll and pitch
    w, x, y, z = torso_quat
    roll = math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = math.asin(2*(w*y - z*x))

    cost += 2.0 * (roll**2 + pitch**2)
    cost += 0.1 * np.sum(qvel[6:]**2)
    cost += 0.01 * np.sum(ctrl**2)

    return cost

def running_cost(x_pos, theta, x_vel, theta_vel, control):
    cart_pos_cost = 1.0 * x_pos**2
    pole_pos_cost = 20.0 * (math.cos(theta) - 1.0)**2
    cart_vel_cost = 0.1 * x_vel**2
    pole_vel_cost = 0.1 * theta_vel**2
    ctrl_cost = 0.01 * control[0]**2
    return cart_pos_cost + pole_pos_cost + cart_vel_cost + pole_vel_cost + ctrl_cost

def terminal_cost(x_pos, theta, x_vel, theta_vel):
    return 10.0 * running_cost(x_pos, theta, x_vel, theta_vel, np.zeros(nu))

# === Control Buffer ===
U_global = np.zeros((nu, T))

# === MPPI Rollout ===
def rollout(model, data, U, noise):
    costs = np.zeros(K)

    for k in range(K):
        d_copy = mujoco.MjData(model)
        d_copy.qpos[:] = data.qpos
        d_copy.qvel[:] = data.qvel
        mujoco.mj_forward(model, d_copy)

        cost = 0.0
        for t in range(T):
            d_copy.ctrl[:] = U[:, t] + noise[:, t, k]
            mujoco.mj_step(model, d_copy)

            x_pos = d_copy.qpos[0]
            theta = d_copy.qpos[1]
            x_vel = d_copy.qvel[0]
            theta_vel = d_copy.qvel[1]

            cost += running_cost(x_pos, theta, x_vel, theta_vel, d_copy.ctrl)

        costs[k] = cost + terminal_cost(
            d_copy.qpos[0], d_copy.qpos[1],
            d_copy.qvel[0], d_copy.qvel[1]
        )

    return costs

# === MPPI Step ===
def mppi_step(model, data):
    noise = np.random.randn(nu, T, K) * sigma
    costs = rollout(model, data, U_global, noise)

    beta = np.min(costs)
    weights = np.exp(-1 / _lambda * (costs - beta))
    weights /= np.sum(weights)

    for t in range(T):
        weighted_sum = sum(weights[k] * noise[:, t, k] for k in range(K))
        U_global[:, t] += weighted_sum

# === Controller Function ===
def mppi_controller(model, data):
    mppi_step(model, data)
    data.ctrl[:] = U_global[:, 0]

    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.1 * U_global[:, -2]  # decay factor

# === Visualize ===
try:
    from mujoco import viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        mppi_controller(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()
finally:
    viewer.close()
