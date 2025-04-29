import mujoco
import numpy as np
import os
from datetime import datetime
import csv

# === Model loading ===
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "cartpole.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# === MPPI Constants ===
K = 75        # sample trajectories
T = 100       # horizon
_lambda = 1.0 # temperature
sigma = 0.75  # exploration noise

nx = model.nq + model.nv
nu = model.nu

# === Logging Setup ===
save_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
SAVE_DIR = os.path.join("data", save_timestamp)
os.makedirs(SAVE_DIR, exist_ok=True)

LOG_STATES = []
LOG_ACTIONS = []
LOG_TIMES = []

def log_data(data, u):
    LOG_TIMES.append(data.time)
    LOG_STATES.append(np.concatenate([data.qpos.copy(), data.qvel.copy()]))
    LOG_ACTIONS.append(u.copy())

# === Cost Function ===
def running_cost(x_pos, theta, x_vel, theta_vel, control):
    cart_pos_cost = 1.0 * x_pos**2
    pole_pos_cost = 20.0 * (np.cos(theta) - 1.0)**2
    cart_vel_cost = 0.1 * x_vel**2
    pole_vel_cost = 0.1 * theta_vel**2
    ctrl_cost = 0.01 * control[0]**2
    return cart_pos_cost + pole_pos_cost + cart_vel_cost + pole_vel_cost + ctrl_cost

def terminal_cost(x_pos, theta, x_vel, theta_vel):
    return 10.0 * running_cost(x_pos, theta, x_vel, theta_vel, np.zeros(nu))

# === MPPI State ===
U_global = np.zeros((nu, T))
d_copies = [mujoco.MjData(model) for _ in range(os.cpu_count())]
temp = np.zeros(nu)

# === Rollout Function ===
def rollout(model, data, U, noise):
    costs = np.zeros(K)

    def simulate_trajectory(k):
        d_copy = d_copies[k % len(d_copies)]
        d_copy.qpos[:] = data.qpos
        d_copy.qvel[:] = data.qvel
        mujoco.mj_forward(model, d_copy)

        cost = 0.0
        for t in range(T):
            d_copy.ctrl[:] = U[:, t] + noise[:, t, k]
            mujoco.mj_step(model, d_copy)

            x_pos, theta = d_copy.qpos
            x_vel, theta_vel = d_copy.qvel
            cost += running_cost(x_pos, theta, x_vel, theta_vel, d_copy.ctrl)

        costs[k] = cost + terminal_cost(*d_copy.qpos, *d_copy.qvel)

    for k in range(K):
        simulate_trajectory(k)

    return costs

# === MPPI Step ===
def mppi_step(model, data):
    noise = np.random.randn(nu, T, K) * sigma
    costs = rollout(model, data, U_global, noise)

    beta = np.min(costs)
    weights = np.exp(-1 / _lambda * (costs - beta))
    weights /= np.sum(weights)
    global temp
    for t in range(T):
        temp.fill(0.0) 
        for k in range(K):
            temp += weights[k] * noise[:, t, k]
        U_global[:, t] += temp

# === MPPI Controller ===
def mppi_controller(model, data):
    mppi_step(model, data)
    data.ctrl[:] = U_global[:, 0]
    log_data(data, U_global[:, 0])

    # Optional goal-switching logic:
    # x_pos = data.qpos[0]
    # if abs(x_pos - goal_pos) < goal_threshold:
    #     goal_counter += 1
    #     goal_pos = ((-1) ** goal_counter) * math.ceil(goal_counter / 2) * goal_step
    #     print(f"Goal Reached {goal_counter} times. New goal: {goal_pos}")

    # Shift controls
    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.1 * U_global[:, -2]

# === Log Saving ===
def save_logs():
    np.savetxt(os.path.join(SAVE_DIR, "states.csv"), np.array(LOG_STATES), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "times.csv"), np.array(LOG_TIMES), delimiter=",")
    print(f"Logs saved to {SAVE_DIR}")

# === Visualization Hook ===
try:
    from mujoco import viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    while viewer.is_running():
        mppi_controller(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()
        log_data(data, data.ctrl)
finally:
    save_logs()
