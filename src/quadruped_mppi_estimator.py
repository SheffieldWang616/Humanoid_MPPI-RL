import mujoco
import numpy as np
import os
import torch
import sys

sys.path.append(".")
from learning.model import FeatureAttentionStatePredictor

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# === Load MuJoCo model ===
model_path = os.path.join(os.path.dirname(__file__), "scene.xml")
mjc_model = mujoco.MjModel.from_xml_path(model_path)
mjc_data = mujoco.MjData(mjc_model)

# === State and action dimensions ===
state_dim = mjc_data.qpos.size + mjc_data.qvel.size
action_dim = mjc_data.ctrl.size

# === Load Trained Network Model ===
net_model = FeatureAttentionStatePredictor(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dim=512,
    num_heads=4,
    attn_layers=2,
    dropout_rate=0.0
).to(device)

pth_path = "checkpoints_quadruped/model_best.pth"
net_model.load_state_dict(torch.load(pth_path, map_location=device))
net_model.eval()

# === MPPI parameters ===
K = 2048
T = 50
_lambda = 10.0
sigma = 0.4
U_global = np.zeros((action_dim, T))

# === Goal Position (world frame) ===
goal_pos = np.array([2.0, 0.0, 0.35])

# === Cost Functions ===
def running_cost(state, control):
    pos = state[:, :3]  # Assume root world position is first 3 dims
    dist_cost = torch.sum((pos - torch.tensor(goal_pos, device=state.device))**2, dim=1)
    ctrl_cost = 0.1 * torch.sum(control**2, dim=1)
    return dist_cost + ctrl_cost

def terminal_cost(state):
    return 10.0 * running_cost(state, torch.zeros_like(state[:, :action_dim]))

# === Rollout using learned model ===
def rollout_learned_model_batched(net_model, state, U, noise, device):
    state_dim = state.shape[0]
    nu, T, K = noise.shape

    x = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).repeat(K, 1)
    noise_K_T_nu = noise.permute(2, 1, 0)
    U_tensor = torch.tensor(U, dtype=torch.float32, device=device)

    costs = torch.zeros(K, device=device)
    for t in range(T):
        u = U_tensor[:, t].unsqueeze(0).repeat(K, 1) + noise_K_T_nu[:, t, :]
        x_in = torch.cat([x, u], dim=1)

        with torch.no_grad():
            delta = net_model(x_in)
            x_next = x + delta

        costs += running_cost(x_next, u)
        x = x_next

    costs += terminal_cost(x)
    return costs

# === MPPI Step ===
def mppi_step(net_model, data):
    global U_global
    state = np.concatenate([data.qpos.copy(), data.qvel.copy()])
    noise = torch.randn(action_dim, T, K, device=device) * sigma

    costs = rollout_learned_model_batched(net_model, state, U_global, noise, device)

    beta = torch.min(costs).item()
    weights = torch.exp(-1 / _lambda * (costs - beta))
    weights /= torch.sum(weights)

    weights = weights.reshape(1, 1, K)
    weighted_noise_sum = torch.sum(noise * weights, dim=2).cpu().numpy()
    U_global = weighted_noise_sum

# === Controller ===
def mppi_controller(net_model, data):
    mppi_step(net_model, data)
    data.ctrl[:] = U_global[:, 0]
    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.1 * U_global[:, -2]

# === Visualization Loop ===
try:
    from mujoco import viewer
    viewer = mujoco.viewer.launch_passive(mjc_model, mjc_data)
    while viewer.is_running():
        mppi_controller(net_model, mjc_data)
        mujoco.mj_step(mjc_model, mjc_data)
        viewer.sync()
finally:
    viewer.close()