import mujoco
import numpy as np
import os
from threading import Thread, Lock
from copy import deepcopy
import mujoco
from mujoco import viewer 
import numpy as np
import os
import time


import os
import time
import numpy as np
from datetime import datetime
import mujoco
from mujoco import MjModel, MjData, mj_step#, mj_loadXML
from mujoco.viewer import launch_passive
import mujoco_viewer as mjv
from pathlib import Path

# === Model and Data Initialization ===
model_path = os.path.join(os.path.dirname(__file__), "humanoid.xml")
model = mujoco.MjModel.from_xml_path(model_path)# mj_loadXML(model_path)
data = MjData(model)

# === MPPI Constants ===
Position = np.array([10.0, 0.0])
K = 30
T = 75
λ = 1.0
Σ = 0.75

nx = model.nq + model.nv
nu = model.nu

# === Logging Setup ===
SAVE_DIR = os.path.join("data", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

LOG_STATES = []
LOG_ACTIONS = []
LOG_TIMES = []

def log_data(d, u):
    LOG_TIMES.append(d.time)
    LOG_STATES.append(np.concatenate([d.qpos.copy(), d.qvel.copy()]))
    LOG_ACTIONS.append(u.copy())

# === Humanoid Cost Function ===
def humanoid_cost(qpos, qvel, ctrl, t):
    cost = 0.0
    root_pos = qpos[:3]
    target_pos = Position

    torso_quat = qpos[3:7]
    root_lin_vel = qvel[:2]
    target_vel = np.array([0.4, 0.0])

    # Orientation: roll, pitch, yaw
    w, x, y, z = torso_quat
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(2 * (w*y - z*x))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))

    cost += 5.0 * (roll**2 + pitch**2)
    cost += 0.1 * yaw**2
    cost += 12.5 * np.linalg.norm(root_pos[:2] - target_pos[:2])
    target_height = 1.28
    cost += 3.0 * (target_height - root_pos[2])
    cost += 1.0 * np.linalg.norm(root_lin_vel - target_vel)

    def get_body_vx(d, body_id):
        return d.cvel[body_id, 0]

    id_left = model.body("shin_left").id
    id_right = model.body("shin_right").id

    vx_left = get_body_vx(data, id_left)
    vx_right = get_body_vx(data, id_right)

    if vx_left > vx_right:
        foot_swing = "foot_left"
        foot_stance = "foot_right"
        knee_swing = id_left
    else:
        foot_swing = "foot_right"
        foot_stance = "foot_left"
        knee_swing = id_right

    swing_id = model.body(foot_swing).id
    stance_id = model.body(foot_stance).id

    swing_foot_x = data.xpos[swing_id, 0]
    foot_targetx = root_pos[0] + 0.5
    cost += 7.5 * (swing_foot_x - foot_targetx)**2

    swing_knee_x = data.xpos[knee_swing, 0]
    cost += 3.5 * (swing_knee_x - foot_targetx)**2

    swing_foot_z = data.xpos[swing_id, 2]
    stance_foot_z = data.xpos[stance_id, 2]
    foot_clearance = swing_foot_z - stance_foot_z
    if foot_clearance < 0:
        cost -= 0.01 * foot_clearance

    left_foot_y = data.xpos[model.body("foot_left").id, 1]
    right_foot_y = data.xpos[model.body("foot_right").id, 1]
    leg_clearance = left_foot_y - right_foot_y
    if leg_clearance < 0:
        cost -= 1.0 * leg_clearance

    cost += 0.01 * np.sum(ctrl ** 2)

    return cost

def terminal_cost(qpos, qvel, t):
    return 10.0 * humanoid_cost(qpos, qvel, np.zeros(nu), t)

# === MPPI Buffers ===
U_global = np.zeros((nu, T))
temp = np.zeros(nu)
d_copies = [MjData(model) for _ in range(os.cpu_count())]

# === MPPI Rollout ===
def rollout(m, d, U, noise):
    costs = np.zeros(K)
    for k in range(K):
        d_copy = d_copies[k % len(d_copies)]
        d_copy.qpos[:] = d.qpos[:]
        d_copy.qvel[:] = d.qvel[:]
        cost = 0.0
        for t in range(T):
            ctrl = U[:, t] + noise[:, t, k]
            d_copy.ctrl[:] = ctrl
            mj_step(m, d_copy)
            cost += humanoid_cost(d_copy.qpos, d_copy.qvel, ctrl, t)
        costs[k] = cost + terminal_cost(d_copy.qpos, d_copy.qvel, T)
    return costs
 
def mppi_step(m, d):
    noise = np.random.randn(nu, T, K) * Σ
    costs = rollout(m, d, U_global, noise)
    β = np.min(costs)
    weights = np.exp(-1 / λ * (costs - β))
    weights /= np.sum(weights)
    global temp
    for t in range(T):
        temp.fill(0.0) #temp[:] = 0.0
        for k in range(K):
            temp += weights[k] * noise[:, t, k]
        U_global[:, t] += temp

def mppi_controller(m, d):
    mppi_step(m, d)
    d.ctrl[:] = U_global[:, 0]
    log_data(d, U_global[:, 0])
    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.1 * U_global[:, -2]

def save_logs():
    np.savetxt(os.path.join(SAVE_DIR, "states.csv"), np.array(LOG_STATES), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "times.csv"), np.array(LOG_TIMES), delimiter=",")
    print(f"Log data saved to: {SAVE_DIR}")

# === Run Visualisation ===

def simulate():
    from mujoco import viewer

    with viewer.launch_passive(model, data) as v:
        print("Simulation running. Close the viewer window to stop.")
        while v.is_running():
            mppi_controller(model, data)
            mujoco.mj_step(model, data)
            v.sync()
            log_data(data, data.ctrl)
            
if __name__ == "__main__":
    simulate()
