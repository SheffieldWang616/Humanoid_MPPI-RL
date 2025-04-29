import numpy as np
import mujoco
import threading
import os
import numpy as np
import mujoco
from joblib import Parallel, delayed
from datetime import datetime
import mujoco
from mujoco import MjModel, MjData, mj_step#, mj_loadXML
from mujoco.viewer import launch_passive
import mujoco_viewer as mjv
from pathlib import Path

# Load model and initialize data
cf = os.path.abspath(__file__)
cwd = os.path.dirname(cf)
model_path = os.path.join(cwd, "scene.xml")
model = mujoco.MjModel.from_xml_path(model_path)
print("Loading model from:", model_path)
data = mujoco.MjData(model)

# Constants
K = 50
H = 30
lam = 0.2
sigma = 0.3
nx = model.nq + model.nv
nu = model.nu
U_global = np.zeros((nu, H))

# Define goal position
# Goal in (x, y)
goal_xy = np.array([2.0, 0.0])
GOAL_TOLERANCE = 1.0

# Leg joint indices (based on standard GO1 layout)
FL_hip, FL_thigh, FL_calf = 0, 1, 2
FR_hip, FR_thigh, FR_calf = 3, 4, 5
RL_hip, RL_thigh, RL_calf = 6, 7, 8
RR_hip, RR_thigh, RR_calf = 9, 10, 11

# === Logging Setup ===
SAVE_DIR = os.path.join("Humanoid_MPPI-RL/data", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

LOG_STATES = []
LOG_ACTIONS = []
LOG_TIMES = []

def log_data(d, u):
    LOG_TIMES.append(d.time)
    LOG_STATES.append(np.concatenate([d.qpos.copy(), d.qvel.copy()]))
    LOG_ACTIONS.append(u.copy())


def cost(qpos, qvel, ctrl):
    global goal_xy
    # Weights
    w_pos = 1000.0
    w_height = 500.0
    w_vel = 20000.0
    w_ori = 500.0
    w_ang = 20.0
    w_ctrl = 0.1
    w_goal = 3000.0
    w_trot = 30000.0#100.0
    w_front = 15000.0
    w_back = 20000.0
    w_knee = 40000.0  # Discourage excessive knee bending
    w_posture = 5.0  # Discourage extreme joint postures
    w_back_leg_symmetry = 50.0  # Encourage symmetry between back legs

    # Targets
    target_height = 0.4  # Target height for body
    target_vel_x = 0.9#0.6  # Target forward velocity (x direction)
    neutral_knee_angle = 0.5  # Nominal knee angle for neutral stance

    # State extraction
    current_pos = qpos[:3]
    current_vel = qvel[:3]
    current_ori = qpos[6:9]
    current_ang = qvel[6:9]
    current_xy = qpos[:2]  # Current x, y position

    # Leg joints
    FL_calf = qpos[2]
    FR_calf = qpos[5]
    RL_calf = qpos[8]  # Left rear calf
    RR_calf = qpos[11]  # Right rear calf

    # Costs
    height_cost = w_height * (current_pos[2] - target_height)**2  # Height tracking
    vel_cost = w_vel * (current_vel[0] - target_vel_x)**2  # Encourage forward velocity in x
    ori_cost = w_ori * (current_ori[0]**2 + current_ori[1]**2)  # Orientation tracking (roll, pitch)
    ang_cost = w_ang * np.sum(current_ang**2)  # Angular velocity cost
    lateral_cost = w_pos * (current_pos[1]**2 + current_vel[1]**2)  # Penalize lateral movement (y direction)
    ctrl_cost = w_ctrl * np.sum(ctrl**2)  # Control cost (penalize large control inputs)
    goal_cost = w_goal * np.sum((current_xy - goal_xy)**2)  # Goal position tracking (2D)

    # Encourage trot gait (symmetry between opposite legs)
    trot_cost = w_trot * ((FL_calf - RR_calf)**2 + (FR_calf - RL_calf)**2)

    # Front leg cost (encourage movement of front legs)
    front_hip_cost = -w_front * (ctrl[1]**2 + ctrl[4]**2)  # Encourage movement of front hips
    front_leg_cost = w_front * (ctrl[2]**2 + ctrl[5]**2)  # Encourage movement of front legs

    # Refined back leg cost (penalize knee bending and encourage symmetry)
    back_hip_cost = -w_back * (ctrl[7]**2 + ctrl[10]**2)  # Encourage movement of back hips
    back_leg_cost = w_back * (ctrl[8]**2 + ctrl[11]**2)  # Encourage movement of back legs

    # Penalize excessive knee bending
    knee_penalty = (
        (FL_calf - neutral_knee_angle)**2 +
        (FR_calf - neutral_knee_angle)**2 +
        (RL_calf - neutral_knee_angle)**2 +
        (RR_calf - neutral_knee_angle)**2
    )
    knee_cost = w_knee * knee_penalty

    # Optional: discourage extreme joint postures (all joints)
    posture_cost = w_posture * np.sum(qpos[0:12]**2)

    # Total cost (sum of all components)
    total_cost = (
        height_cost + vel_cost + ori_cost + ang_cost +
        lateral_cost + ctrl_cost + goal_cost +
        trot_cost + front_leg_cost + back_leg_cost + knee_cost +
        posture_cost + front_hip_cost + back_hip_cost
    )

    # **Encourage forward movement by optimizing velocity in the x direction**
    # You could also reward velocity in the x direction more directly like this:
    forward_motion_cost = w_vel * (current_vel[0])**2  # Reward forward movement along x-axis
    total_cost += forward_motion_cost

    return total_cost


def rollout(model, data, U, noise):
    costs = np.zeros(K)

    def worker(k):
        d_copy = mujoco.MjData(model)
        np.copyto(d_copy.qpos, data.qpos)
        np.copyto(d_copy.qvel, data.qvel)
        cost_sum = 0.0
        for t in range(H):
            current_ctrl = U[:, t] + noise[:, t, k]
            d_copy.ctrl[:] = np.clip(current_ctrl, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, d_copy)
            cost_sum += cost(d_copy.qpos, d_copy.qvel, d_copy.ctrl)
        costs[k] = cost_sum

    threads = []
    for k in range(K):
        t = threading.Thread(target=worker, args=(k,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    return costs

def mppi_update(model, data):
    global U_global
    noise = np.random.randn(nu, H, K) * sigma
    costs = rollout(model, data, U_global, noise)
    data.ctrl[:] = U_global[:, 0]
    log_data(data, U_global[:, 0])

    beta = np.min(costs)
    weights = np.exp(-1 / lam * (costs - beta))
    weights /= np.sum(weights) + 1e-10

    for t in range(H):
        weighted_noise = np.sum(weights[k] * noise[:, t, k] for k in range(K))
        U_global[:, t] = np.clip(
            U_global[:, t] + weighted_noise,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1]
        )

    #data.ctrl[:] = U_global[:, 0]
    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.0



def save_logs():
    np.savetxt(os.path.join(SAVE_DIR, "states.csv"), np.array(LOG_STATES), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "times.csv"), np.array(LOG_TIMES), delimiter=",")
    print(f"Log data saved to: {SAVE_DIR}")

# Visualization
# Simulation loop
from mujoco import viewer

def simulate():
    from mujoco import viewer
    
    goal_reached = False

    with viewer.launch_passive(model, data) as v:
        print("Simulation running. Close the viewer window to stop.")
        while v.is_running():
            mppi_update(model, data)
            mujoco.mj_step(model, data)
            v.sync()
            log_data(data, data.ctrl)
            global goal_xy
            
            dist_to_goal = np.linalg.norm(data.qpos[:2] - goal_xy)
            if dist_to_goal < GOAL_TOLERANCE:
                goal_reached = True
                print("Goal reached!")
                break
            if data.qpos[1] >= goal_xy[0]:
                goal_reached = True
                print("Goal reached!")
                break

    if goal_reached:
        save_logs()
    #save_logs()

# Main
if __name__ == "__main__":
    simulate()




'''
def cost(qpos, qvel, ctrl):
    global goal_xy
    # Weights
    w_pos = 1000.0
    w_height = 500.0
    w_vel = 1000.0
    w_ori = 500.0
    w_ang = 20.0
    w_ctrl = 0.1
    w_goal = 3000.0
    w_trot = 100.0
    w_front = 50000.0 #50000?
    w_back = 50000.0 #50000?
    w_knee = 5000.0     # New: discourage excessive knee bending
    w_posture = 5.0    # Optional: discourage extreme joint postures

    # Targets
    target_height = 0.35#0.45
    target_vel_x = 0.6
    neutral_knee_angle = -1.5#-1.3  # GO1-like nominal knee bend
    # -4.0
    # 0.5 is pretty good, - - - -

    # State extraction
    current_pos = qpos[:3]
    print(current_pos)
    current_vel = qvel[:3]
    current_ori = qpos[6:9]
    current_ang = qvel[6:9]
    current_xy = qpos[:2]

    # Leg joints
    FL_calf = qpos[2]
    FR_calf = qpos[5]
    RL_calf = qpos[8]
    RR_calf = qpos[11]

    # Costs
    height_cost = w_height * (current_pos[2] - target_height)**2
    vel_cost = w_vel * (current_vel[0] - target_vel_x)**2
    ori_cost = w_ori * (current_ori[0]**2 + current_ori[1]**2)
    ang_cost = w_ang * np.sum(current_ang**2)
    lateral_cost = w_pos * (current_pos[1]**2 + current_vel[1]**2)
    ctrl_cost = w_ctrl * np.sum(ctrl**2)
    goal_cost = w_goal * np.sum((current_xy - goal_xy)**2)

    # Trot gait encouragement
    trot_cost = w_trot * ((FL_calf - RR_calf)**2 + (FR_calf - RL_calf)**2)
    front_leg_cost = w_front * (ctrl[2]**2 + ctrl[5]**2)
    back_leg_cost = w_back * (ctrl[2]**2 + ctrl[5]**2)

    # Penalize excessive knee bending
    knee_penalty = (
        (FL_calf - neutral_knee_angle)**2 +
        (FR_calf - neutral_knee_angle)**2 +
        (RL_calf - neutral_knee_angle)**2 +
        (RR_calf - neutral_knee_angle)**2
    )
    knee_cost = w_knee * knee_penalty

    # Optional: discourage extreme joint postures (all joints)
    posture_cost = w_posture * np.sum(qpos[0:12]**2)

    total_cost = (
        height_cost + vel_cost + ori_cost + ang_cost +
        lateral_cost + ctrl_cost + goal_cost +
        trot_cost + front_leg_cost + back_leg_cost + knee_cost + posture_cost
    )
    return total_cost
'''







'''
import numpy as np
import mujoco
from mujoco import MjModel, MjData, mj_step#, mj_loadXML
from mujoco.viewer import launch_passive
import mujoco_viewer as mjv
from joblib import Parallel, delayed
import os
from pathlib import Path
from datetime import datetime

# Paths
cf = os.path.abspath(__file__)
cwd = os.path.dirname(cf)
model_path = os.path.join(cwd, "scene.xml")
print("Loading model from:", model_path)

# Load model and data
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

# Constants
K = 50        # Number of samples
H = 30        # Horizon length
lam = 0.2     # Temperature λ
sigma = 0.3   # Noise std
nx = model.nq + model.nv
nu = model.nu

U_global = np.zeros((nu, H))
U_global[2::3, :] = 0.1  # Initial knee flexion (helpful!)

# Save directory
SAVE_DIR = os.path.join("Humanoid_MPPI-RL/data", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

LOG_STATES = []
LOG_ACTIONS = []
LOG_TIMES = []

# Goal (x, y)
goal_position = np.array([4.0, 0.0])

# Logging function
def log_data(d, u):
    LOG_TIMES.append(d.time)
    LOG_STATES.append(np.concatenate([d.qpos.copy(), d.qvel.copy()]))
    LOG_ACTIONS.append(u.copy())

# Cost function
def cost(qpos, qvel, ctrl, goal):
    # Weights
    w_pos = 1000.0    
    w_height = 500.0  
    w_vel = 1000.0    
    w_ori = 500.0     
    w_ang = 20.0      
    w_ctrl = 0.1      
    w_goal = 3000.0   
    w_forward = 500.0   # NEW: reward forward motion

    target_height = 0.45

    current_pos = qpos[:3]
    current_vel = qvel[:3]
    current_ori = qpos[6:9]
    current_ang = qvel[6:9]

    # Cost components
    height_cost = w_height * (current_pos[2] - target_height)**2
    sideways_vel_cost = w_vel * (current_vel[1])**2
    ori_cost = w_ori * (current_ori[0]**2 + current_ori[1]**2)
    ang_cost = w_ang * np.sum(current_ang**2)
    lateral_cost = w_pos * (current_pos[1]**2 + current_vel[1]**2)
    ctrl_cost = w_ctrl * np.sum(ctrl**2)
    goal_cost = w_goal * np.sum((current_pos[:2] - goal)**2)

    # Forward velocity reward
    delta_to_goal = goal - current_pos[:2]
    direction = delta_to_goal / (np.linalg.norm(delta_to_goal) + 1e-6)  # Unit vector
    forward_vel = np.dot(current_vel[:2], direction)
    forward_vel_reward = -w_forward * (forward_vel - 0.5)**2  # Prefer ~0.5 m/s

    total_cost = (
        height_cost +
        sideways_vel_cost +
        ori_cost +
        ang_cost +
        lateral_cost +
        ctrl_cost +
        goal_cost +
        forward_vel_reward
    )

    return total_cost

# Single trajectory simulation
def simulate_single_trajectory(U, noise, model, data, goal):
    d_copy = mujoco.MjData(model)
    np.copyto(d_copy.qpos, data.qpos)
    np.copyto(d_copy.qvel, data.qvel)
    cost_sum = 0.0

    for t in range(H):
        ctrl = np.clip(U[:, t] + noise[:, t], -10.0, 10.0)
        d_copy.ctrl[:] = ctrl
        mujoco.mj_step(model, d_copy)
        cost_sum += cost(d_copy.qpos, d_copy.qvel, d_copy.ctrl, goal)

    return cost_sum

# Rollout over all K samples
def rollout(model, data, U, noise, goal):
    costs = Parallel(n_jobs=-1)(
        delayed(simulate_single_trajectory)(U, noise[:, :, k], model, data, goal)
        for k in range(K)
    )
    return np.array(costs)

# MPPI update
def mppi_update(model, data):
    global U_global
    noise = np.random.randn(nu, H, K) * sigma
    costs = rollout(model, data, U_global, noise, goal_position)

    beta = np.min(costs)
    weights = np.exp(-1.0 / lam * (costs - beta))
    weights /= np.sum(weights) + 1e-10

    weighted_noise = np.tensordot(noise, weights, axes=(2, 0))
    U_global += weighted_noise
    U_global = np.clip(U_global, -10.0, 10.0)

    data.ctrl[:] = U_global[:, 0]
    U_global[:, :-1] = U_global[:, 1:]
    U_global[:, -1] = 0.0

# Save logs
def save_logs():
    np.savetxt(os.path.join(SAVE_DIR, "states.csv"), np.array(LOG_STATES), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "times.csv"), np.array(LOG_TIMES), delimiter=",")
    print(f"Log data saved to: {SAVE_DIR}")

# Simulation loop
def simulate():
    from mujoco import viewer

    with viewer.launch_passive(model, data) as v:
        print("Simulation running. Close the viewer window to stop.")
        while v.is_running():
            mppi_update(model, data)
            mj_step(model, data)
            v.sync()
            #log_data(data, data.ctrl)

    #save_logs()

# Main
if __name__ == "__main__":
    simulate()

'''
