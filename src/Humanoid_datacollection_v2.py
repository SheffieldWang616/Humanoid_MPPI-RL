# Based on Humanoid_datacollection_v2.jl
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

# Simulation timestep (fixed)
dt = model.opt.timestep

# === MPPI Constants ===
Position = np.array([2.0, 0.0, 1.28])
goal_step = np.array([2.0, 0.0, 0.0])
goal_counter = 0
goal_threshold = 0.15
K = 30
T = 75
λ = 1.0
Σ = 0.75

nx = model.nq + model.nv
nu = model.nu

# how many consecutive frames to wait before switching
PHASE_DELAY = 2

# last instantaneously‐detected side ("left" or "right")
_last_inst_side = None  

# how many times in a row we've seen _last_inst_side
_inst_count     = 0     

# which side we're actually committed to swinging
_committed_side = "left"  # or whichever you want as your default

# === Logging Setup ===
SAVE_DIR = os.path.join("data", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

LOG_STATES = []
LOG_ACTIONS = []
LOG_TIMES = []

# === New: finite-difference velocity for states ===
prev_qpos = None

def log_data(d: MjData, u: np.ndarray):
    global prev_qpos

    # compute velocity estimate from position
    if prev_qpos is None:
        vel = np.zeros_like(d.qpos)
    else:
        vel = (d.qpos - prev_qpos) / dt

    # update for next call
    prev_qpos = d.qpos.copy()

    # log time, state (qpos + vel_est), and action
    LOG_TIMES.append(d.time)
    LOG_STATES.append(np.concatenate([d.qpos.copy(), vel]))
    LOG_ACTIONS.append(u.copy())

# === Humanoid Cost Function ===
def humanoid_cost(qpos, qvel, bvel, ctrl, t):
    cost = 0.0
    root_pos = qpos[:3]
    global Position
    target_pos = Position

    torso_quat = qpos[3:7]
    root_lin_vel = qvel[:2]
    target_vel = np.array([0.3, 0.0])

    # Orientation: roll, pitch, yaw
    w, x, y, z = torso_quat
    roll = np.arctan2(2 * (w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(2 * (w*y - z*x))
    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2*(y*y + z*z))

    cost += 4.0 * (roll**2 + pitch**2)
    cost += 0.5 * yaw**2
    cost += 3.0 * np.linalg.norm(root_pos[:2] - target_pos[:2])**2
    target_height = 1.28
    cost += 10.0 * abs(target_height - root_pos[2])
    cost += 2.0 * abs(root_pos[1])
    #cost += 1.0 * np.linalg.norm(root_lin_vel - target_vel)

    def get_body_vx(d, body_id):
        return bvel[body_id, 0]
    
    # helper to grab a joint velocity by name
    def get_joint_vel(qvel, joint_name):
        joint_id = model.joint(joint_name).id
        return qvel[joint_id]

    id_left = model.body("shin_left").id
    id_right = model.body("shin_right").id

    vx_left  = get_joint_vel(qvel, "ankle_x_left")
    vx_right = get_joint_vel(qvel, "ankle_x_right")

    #if vx_left > vx_right:
    #    foot_swing = "foot_left"
    #    foot_stance = "foot_right"
    #    knee_swing = id_left
    #else:
    #    foot_swing = "foot_right"
    #    foot_stance = "foot_left"
    #    knee_swing = id_right

        # identify which foot is momentarily higher
    left_higher = data.xpos[ model.body("foot_left").id, 2 ] \
                > data.xpos[ model.body("foot_right").id, 2 ]
    #inst_side = "left" if left_higher else "right"
    inst_side = "left" if vx_left > vx_right else "right"

    global _last_inst_side, _inst_count, _committed_side

    # if we saw the same side as last frame, bump the counter,
    # otherwise reset
    if inst_side == _last_inst_side:
        _inst_count += 1
    else:
        _last_inst_side = inst_side
        _inst_count     = 1

    # only once we’ve seen the same side PHASE_DELAY frames in a row
    # do we actually switch our committed swing foot
    if _inst_count >= PHASE_DELAY:
        _committed_side = inst_side

    # now set foot_swing/knee based on the *committed* choice
    if _committed_side == "left":
        foot_swing  = "foot_left"
        foot_stance = "foot_right"
        knee_swing  = model.body("shin_left").id
    else:
        foot_swing  = "foot_right"
        foot_stance = "foot_left"
        knee_swing  = model.body("shin_right").id

    swing_id = model.body(foot_swing).id
    stance_id = model.body(foot_stance).id

    # assume root_body is the name of your root link, e.g. "torso"
    root_body_id = model.body("torso").id

    # 1) extract the body→world rotation matrix
    R = data.xmat[root_body_id].reshape(3, 3)  

    # 2) the first column of R is your body‐X axis in world frame
    forward_world = R[:, 0]   # unit vector pointing “forward”

    # 3) project both root and swing foot positions onto that axis
    root_proj   = forward_world.dot(root_pos)
    swing_proj  = forward_world.dot(data.xpos[swing_id])

    # 4) compare to desired offset (0.5 m ahead of the root)
    desired_proj = root_proj + 0.3
    cost += 8.0 * abs(swing_proj - desired_proj)

    # Reward positive knee‐joint velocity (only when it’s > 0):
    knee_vel = get_joint_vel(qvel, swing_id)   # or "knee_right"
    if knee_vel > 0:
        cost -= 0.2 * knee_vel
    else:
        # Optional: penalize if it moves backward
        cost += 0.05 * (-knee_vel)

    swing_knee_x = forward_world.dot(data.xpos[knee_swing])
    cost += 3.0 * abs(swing_knee_x - desired_proj)

    swing_foot_z = data.xpos[swing_id, 2]
    stance_foot_z = data.xpos[stance_id, 2]
    cost += 0.005 * np.abs(stance_foot_z)
    
    foot_clearance = swing_foot_z - stance_foot_z
    #if foot_clearance < 0.05:
    #    cost += 2.0 * foot_clearance**2

    left_foot_y = data.xpos[model.body("foot_left").id, 1]
    right_foot_y = data.xpos[model.body("foot_right").id, 1]
    leg_clearance = left_foot_y - right_foot_y
    if leg_clearance < 0.05:
        #cost -= 1.0 * leg_clearance
        cost += 1.0 * leg_clearance**2

    cost += 0.01 * np.sum(ctrl ** 2)

    return cost

def terminal_cost(qpos, qvel, bvel, t):
    return 10.0 * humanoid_cost(qpos, qvel, bvel, np.zeros(nu), t)

# === MPPI Buffers & Helpers ===
U_global    = np.zeros((nu, T))
d_copies    = [MjData(model) for _ in range(os.cpu_count())]
# for joint-vel masking
prev_qpos_copies   = [None] * len(d_copies)
# for body-COM-vel masking
prev_xpos_copies   = [None] * len(d_copies)

temp        = np.zeros(nu)

# === MPPI Rollout (with masked qvel) ===
def rollout(m, d, U, noise):
    # reset histories
    for i in range(len(d_copies)):
        prev_qpos_copies[i] = None
        prev_xpos_copies[i] = None

    costs = np.zeros(K)

    # single loop over all (trajectory, time) pairs
    for kt in range(K * T):
        k, t = divmod(kt, T)           # k in [0..K), t in [0..T)
        d_copy = d_copies[k]
        # copy the “root” state at the start of each trajectory
        if t == 0:
            d_copy.qpos[:] = d.qpos[:]
            d_copy.qvel[:] = d.qvel[:]

        # apply control + noise
        ctrl = U[:, t] + noise[:, t, k]
        d_copy.ctrl[:] = ctrl
        mj_step(m, d_copy)

        # 1) joint-vel mask
        if prev_qpos_copies[k] is None:
            vel_q = np.zeros_like(d_copy.qpos)
        else:
            vel_q = (d_copy.qpos - prev_qpos_copies[k]) / dt
        prev_qpos_copies[k] = d_copy.qpos.copy()

        # 2) body-COM-vel mask
        if prev_xpos_copies[k] is None:
            vel_body = np.zeros_like(d_copy.xpos)
        else:
            vel_body = (d_copy.xpos - prev_xpos_copies[k]) / dt
        prev_xpos_copies[k] = d_copy.xpos.copy()

        # accumulate cost for this (k,t)
        costs[k] += humanoid_cost(d_copy.qpos, vel_q, vel_body, ctrl, t)

    # terminal cost (once per trajectory)
    for k in range(K):
        d_copy = d_copies[k]
        costs[k] += terminal_cost(
            d_copy.qpos,
            np.zeros_like(d_copy.qpos),
            np.zeros_like(d_copy.xpos),
            T
        )
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

'''
def mppi_controller(m, d):
    mppi_step(m, d)
    d.ctrl[:] = U_global[:, 0]
    log_data(d, U_global[:, 0])
    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.1 * U_global[:, -2]
'''
def mppi_controller(model, data):
    global Position, goal_counter

    mppi_step(model, data)
    data.ctrl[:] = U_global[:, 0]
    log_data(data, U_global[:, 0])

    # check & update goal *before* sampling
    root = data.qpos[:3]
    if np.linalg.norm(root-Position) < goal_threshold:
        goal_counter += 1
        Position += goal_step
        print(f"Goal Reached {goal_counter}. New goal: {Position}")

    # now plan with the up-to-date target
    mppi_step(model, data)
    data.ctrl[:] = U_global[:,0]
    log_data(data, data.ctrl)
    # …rest of your shifting U_global buffer…

    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.1 * U_global[:, -2]

def save_logs():
    np.savetxt(os.path.join(SAVE_DIR, "states.csv"), np.array(LOG_STATES), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "times.csv"), np.array(LOG_TIMES), delimiter=",")
    print(f"Log data saved to: {SAVE_DIR}")

# === Run Visualisation ===

def simulate():

    with viewer.launch_passive(model, data) as v:
        print("Simulation running. Close the viewer window to stop.")
        while v.is_running():
            mppi_controller(model, data)
            mujoco.mj_step(model, data)
            v.sync()
            log_data(data, data.ctrl)
            
if __name__ == "__main__":
    try:
        simulate()
    finally:
        save_logs()




'''
# Based on Humanoid_datacollection.jl
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
Position = np.array([4.0, 0.0])
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
    global Position
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
'''