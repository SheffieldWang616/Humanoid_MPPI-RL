import mujoco
import numpy as np
import math
import os

import torch
import sys
sys.path.append("/home/yaguangl/Desktop/Humanoid_MPPI-RL-main/Humanoid_MPPI-RL-main")

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from learning.model import FeatureAttentionStatePredictor

# === Load MuJoCo model for visualization only ===
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "cartpole.xml")
mjc_model = mujoco.MjModel.from_xml_path(model_path)
mjc_data = mujoco.MjData(mjc_model)

# === Load Trained Network Model ===
net_model = FeatureAttentionStatePredictor(
    state_dim=4, action_dim=1, hidden_dim=64, num_heads=4, attn_layers=2, dropout_rate=0.0
).to(device)

pth_path = "/home/yaguangl/Desktop/Humanoid_MPPI-RL-main/Humanoid_MPPI-RL-main/checkpoints_cartpole/model_final.pth"
net_model.load_state_dict(torch.load(pth_path, map_location=device))
net_model.eval()

# === MPPI constants ===
K = 32
T = 100
_lambda = 1.0
sigma = 1.0

nx = 4  # [x, theta, dx, dtheta]
nu = 1

# === Costs (unchanged)===
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

# === Rollout using learned model ===
def rollout_learned_model_batched(net_model, state, U, noise, device):
    # state: (state_dim,)
    # U: (nu, T)
    # noise: (nu, T, K)
    # Returns: costs: (K,)

    state_dim = state.shape[0]
    nu, T, K = noise.shape

    # Initial state: repeat for K trajectories
    x = np.repeat(state[None, :], K, axis=0)  # (K, state_dim)
    costs = np.zeros(K)

    # np: (nu, T, K) --> (K, T, nu):
    noise_K_T_nu = noise.transpose(2, 1, 0)  # (K, T, nu)

    for t in range(T):
        u = (U[:, t].reshape(1, nu) + noise_K_T_nu[:, t, :])        # (K, nu)
        u = np.clip(u, -1.0, 1.0)

        # Prepare batched input (K, state_dim + action_dim)
        x_in = np.concatenate([x, u], axis=1)
        x_in_tensor = torch.from_numpy(x_in).float().to(device)

        with torch.no_grad():
            x_next = net_model(x_in_tensor).cpu().numpy()  # (K, state_dim)

        # Cost - vectorized
        x_pos, theta, x_vel, theta_vel = x_next[:, 0], x_next[:, 1], x_next[:, 2], x_next[:, 3]
        ctrl = u

        costs += (
            1.0 * x_pos**2 +
            20.0 * (np.cos(theta) - 1.0)**2 +
            0.1 * x_vel**2 +
            0.1 * theta_vel**2 +
            0.01 * ctrl[:, 0]**2
        )

        x = x_next  # Propagate

    # Vectorized terminal cost
    x_pos, theta, x_vel, theta_vel = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    costs += 10.0 * (
        1.0 * x_pos**2 +
        20.0 * (np.cos(theta) - 1.0)**2 +
        0.1 * x_vel**2 +
        0.1 * theta_vel**2
    )

    return costs

# === MPPI Step with learned dynamics ===
def mppi_step(net_model, data):
    state = np.concatenate([data.qpos.copy(), data.qvel.copy()])    # shape (4,)
    noise = np.random.randn(nu, T, K) * sigma
    costs = rollout_learned_model_batched(net_model, state, U_global, noise, device)

    beta = np.min(costs)
    weights = np.exp(-1 / _lambda * (costs - beta))
    weights = weights / np.sum(weights)

    for t in range(T):
        weighted_sum = sum(weights[k] * noise[:, t, k] for k in range(K))
        U_global[:, t] += weighted_sum

# === Controller uses learned model to update actions, environment to step ===
def mppi_controller(net_model, data):
    mppi_step(net_model, data)
    data.ctrl[:] = U_global[:, 0]
    print("Current Control:", data.ctrl[:])  # Debugging line to check control values
    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.1 * U_global[:, -2]  # decay factor

# === Visualization (env still steps physical model for display) ===
try:
    from mujoco import viewer
    viewer = mujoco.viewer.launch_passive(mjc_model, mjc_data)
    while viewer.is_running():
        mppi_controller(net_model, mjc_data)
        print("mjc_data.crtl:", mjc_data.ctrl[:])
        print("mjc_data.qpos:", mjc_data.qpos[:])
        print("mjc_data.qvel:", mjc_data.qvel[:])
        mujoco.mj_step(mjc_model, mjc_data)
        viewer.sync()
finally:
    viewer.close()
