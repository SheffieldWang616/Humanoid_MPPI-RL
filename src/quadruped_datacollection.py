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

def cost(qpos, qvel, ctrl, time):
    global goal_xy

    # Parameters
    trot_period = 0.5  # seconds per trot cycle
    phase = (time % trot_period) / trot_period * 2 * np.pi
    trot_symmetry = np.sin(phase)

    # Weights (tuned)
    w_pos = 50000.0 #40000.0
    w_height = 500.0
    w_vel = 30000.0
    w_ori = 500.0
    w_ang = 20.0
    w_ctrl = 0.01  # Lower to allow exploration
    w_goal = 3000.0
    w_trot = 34000.0
    w_front = 4400.0#4400.0
    w_back = 10000.0#10000.0
    w_knee = 2000.0
    w_posture = 5.0
    w_back_leg_symmetry = 50.0

    # Targets
    target_height = 0.4
    base_target_vel_x = 0.9
    osc_amp = 0.1
    target_vel_x = base_target_vel_x + osc_amp * np.sin(phase)
    neutral_knee_angle = 0.5

    # State extraction
    current_pos = qpos[:3]
    current_vel = qvel[:3]
    current_ori = qpos[6:9]
    current_ang = qvel[6:9]
    current_xy = qpos[:2]

    # Leg joints
    FL_calf = qpos[2]
    FR_calf = qpos[5]
    RL_calf = qpos[8]
    RR_calf = qpos[11]

    # Base costs
    height_cost = w_height * (current_pos[2] - target_height)**2
    vel_cost = w_vel * (current_vel[0] - target_vel_x)**2
    ori_cost = w_ori * (current_ori[0]**2 + current_ori[1]**2)
    ang_cost = w_ang * np.sum(current_ang**2)
    lateral_cost = w_pos * (current_pos[1]**2 + current_vel[1]**2)
    ctrl_cost = w_ctrl * np.sum(ctrl**2)
    goal_cost = w_goal * np.sum((current_xy - goal_xy)**2)

    # Gait symmetry (phase-based)
    FL_RR_phase = (FL_calf - RR_calf) * trot_symmetry
    FR_RL_phase = (FR_calf - RL_calf) * -trot_symmetry
    trot_phase_cost = w_trot * (FL_RR_phase**2 + FR_RL_phase**2)

    # Leg-specific movement encouragement
    front_hip_cost = -w_front * (ctrl[1]**2 + ctrl[4]**2)
    front_leg_cost = w_front * (ctrl[2]**2 + ctrl[5]**2)
    back_hip_cost = -w_back * (ctrl[7]**2 + ctrl[10]**2)
    back_leg_cost = w_back * (ctrl[8]**2 + ctrl[11]**2)

    # Joint posture & symmetry
    knee_penalty = (
        (FL_calf - neutral_knee_angle)**2 +
        (FR_calf - neutral_knee_angle)**2 +
        (RL_calf - neutral_knee_angle)**2 +
        (RR_calf - neutral_knee_angle)**2
    )
    knee_cost = w_knee * knee_penalty
    posture_cost = w_posture * np.sum(qpos[0:12]**2)

    # Total cost
    total_cost = (
        height_cost + vel_cost + ori_cost + ang_cost +
        lateral_cost + ctrl_cost + goal_cost +
        trot_phase_cost + front_leg_cost + back_leg_cost +
        knee_cost + posture_cost + front_hip_cost + back_hip_cost
    )

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
            cost_sum += cost(d_copy.qpos, d_copy.qvel, d_copy.ctrl, d_copy.time)
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



SAVE_DIR = os.path.join("Humanoid_MPPI-RL/quad_data", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)

def save_logs():
    np.savetxt(os.path.join(SAVE_DIR, "states.csv"), np.array(LOG_STATES), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
    np.savetxt(os.path.join(SAVE_DIR, "times.csv"), np.array(LOG_TIMES), delimiter=",")
    print(f"Log data saved to: {SAVE_DIR}")

# Visualization
# Simulation loop
from mujoco import viewer


GOAL_TOLERANCE = 0.5
        
def simulate():
    from mujoco import viewer
    
    global goal_xy, LOG_STATES, LOG_ACTIONS, LOG_TIMES, U_global

    goal_reached = True
    n_runs = 100
    for i in range(n_runs):
        goal_xy = np.array([i + 2.0, 0.0]) #0.01 * (i + 200),
        print(f"\n=== Run {i+1}/{n_runs} | Goal: {goal_xy} ===")

        # Reset simulation state
        mujoco.mj_resetData(model, data)
        U_global = np.zeros((nu, H))  # Reset control sequence
        LOG_STATES = []
        LOG_ACTIONS = []
        LOG_TIMES = []
    
        with viewer.launch_passive(model, data) as v:
            print("Simulation running. Close the viewer window to stop.")
            while v.is_running():
                mppi_update(model, data)
                mujoco.mj_step(model, data)
                v.sync()
                log_data(data, data.ctrl)
                
                dist_to_goal = np.linalg.norm(data.qpos[:2] - goal_xy)
                if (data.qpos[2] < 0.08):
                    print(f"Robot too low")
                    goal_reached = False
                    break
                if dist_to_goal < GOAL_TOLERANCE:
                    goal_reached = True
                    print("Goal reached!")
                    SAVE_DIR = os.path.join("Humanoid_MPPI-RL/quad_data_goal", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
                    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
                    run_dir = os.path.join(SAVE_DIR, f"run_{i:03d}")
                    Path(run_dir).mkdir(parents=True, exist_ok=True)
                    np.savetxt(os.path.join(run_dir, "states.csv"), np.array(LOG_STATES), delimiter=",")
                    np.savetxt(os.path.join(run_dir, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
                    np.savetxt(os.path.join(run_dir, "times.csv"), np.array(LOG_TIMES), delimiter=",")
                    break
                if data.qpos[0] >= goal_xy[0]:
                    #print(data.qpos[0], data.qpos[0])
                    goal_reached = True
                    print("Goal Line (X) reached!")
                    SAVE_DIR = os.path.join("Humanoid_MPPI-RL/quad_data_goal_line", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
                    Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
                    run_dir = os.path.join(SAVE_DIR, f"run_{i:03d}")
                    Path(run_dir).mkdir(parents=True, exist_ok=True)
                    np.savetxt(os.path.join(run_dir, "states.csv"), np.array(LOG_STATES), delimiter=",")
                    np.savetxt(os.path.join(run_dir, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
                    np.savetxt(os.path.join(run_dir, "times.csv"), np.array(LOG_TIMES), delimiter=",")
                    break  


# Main
if __name__ == "__main__":
    simulate()