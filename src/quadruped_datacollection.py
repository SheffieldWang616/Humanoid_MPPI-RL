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


def cost(qpos, qvel, ctrl):
    global goal_xy
    # Weights
    w_pos = 50000.0# 1000.0
    w_height = 5000.0
    w_vel = 24000.0
    w_ori = 500.0
    w_ang = 20.0
    w_ctrl = 0.1
    w_goal = 20000.0
    w_trot = 30000.0#30000.0#100.0
    w_front = 40000.0
    w_back = 50000.0
    w_knee = 40000.0  # Discourage excessive knee bending
    w_posture = 5.0  # Discourage extreme joint postures

    # Targets
    target_height = 0.36  # Target height for body
    target_vel_x = 0.7#0.6  # Target forward velocity (x direction)
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
    
    RL_thigh, RL_calf = 7, 8
    RR_thigh, RR_calf = 10, 11


    # Encourage front legs to stay in front of back legs
    '''w_cross = 20000.0
    leg_crossing_cost = w_cross * (
        np.maximum(0, RL_x - FL_x)**2 +  # Rear left ahead of front left
        np.maximum(0, RR_x - FR_x)**2    # Rear right ahead of front right
    )
    '''

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
    
    w_back_leg_asymmetry = 40000.0
    back_leg_symmetry_cost = w_back_leg_asymmetry * (RL_calf - RR_calf)**2
    total_cost += back_leg_symmetry_cost
    
    w_front_leg_asymmetry = 30000.0
    front_leg_symmetry_cost = w_front_leg_asymmetry * (FL_calf - FR_calf)**2
    total_cost += front_leg_symmetry_cost

    
    # Set a threshold for extension, e.g., -0.4 radians
    max_extension = -0.4
    w_back_extension = 10000.0
    back_leg_extension_cost = w_back_extension * (
        np.maximum(0, max_extension - qpos[RL_thigh])**2 +
        np.maximum(0, max_extension - qpos[RR_thigh])**2
    )
    total_cost += back_leg_extension_cost
    
    max_extension = 0.4
    w_front_extension = 20000.0
    front_leg_extension_cost = w_front_extension * (
        np.maximum(0, max_extension - qpos[FL_thigh])**2 +
        np.maximum(0, max_extension - qpos[FR_thigh])**2
    )
    total_cost += front_leg_extension_cost
    

    # **Encourage forward movement by optimizing velocity in the x direction**
    # You could also reward velocity in the x direction more directly like this:
    #forward_motion_cost = w_vel * (current_vel[0])**2  # Reward forward movement along x-axis
    #total_cost += forward_motion_cost
    

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


def simulate_headless_n_runs(n_runs=100):
    global goal_xy, LOG_STATES, LOG_ACTIONS, LOG_TIMES, U_global

    goal_reached = True
    for i in range(n_runs):
        goal_xy = np.array([0.5 * (i + 4), 0.0])
        print(f"\n=== Run {i+1}/{n_runs} | Goal: {goal_xy} ===")

        # Reset simulation state
        mujoco.mj_resetData(model, data)
        U_global = np.zeros((nu, H))  # Reset control sequence
        LOG_STATES = []
        LOG_ACTIONS = []
        LOG_TIMES = []
            
        step = 0
        while True:
            mppi_update(model, data)
            mujoco.mj_step(model, data)
            log_data(data, data.ctrl)

            step += 1
            dist_to_goal = np.linalg.norm(data.qpos[:2] - goal_xy)
            if (data.qpos[2] < 0.15):
                print(f"Run {i+1}: Robot too low")
                goal_reached = False
                break
            if dist_to_goal < GOAL_TOLERANCE:
                print(f"Run {i+1}: Goal reached in {step} steps")
                SAVE_DIR = os.path.join("Humanoid_MPPI-RL/quad_data_goal", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
                Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
                run_dir = os.path.join(SAVE_DIR, f"run_{i:03d}")
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                np.savetxt(os.path.join(run_dir, "states.csv"), np.array(LOG_STATES), delimiter=",")
                np.savetxt(os.path.join(run_dir, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
                np.savetxt(os.path.join(run_dir, "times.csv"), np.array(LOG_TIMES), delimiter=",")
                print(f"Log data saved to: {SAVE_DIR}")
                break
            if data.qpos[0] >= goal_xy[0]:
                print(f"Run {i+1}: Passed goal x-line in {step} steps")
                SAVE_DIR = os.path.join("Humanoid_MPPI-RL/quad_data_goal_line", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
                Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
                run_dir = os.path.join(SAVE_DIR, f"run_{i:03d}")
                Path(run_dir).mkdir(parents=True, exist_ok=True)
                np.savetxt(os.path.join(run_dir, "states.csv"), np.array(LOG_STATES), delimiter=",")
                np.savetxt(os.path.join(run_dir, "actions.csv"), np.array(LOG_ACTIONS), delimiter=",")
                np.savetxt(os.path.join(run_dir, "times.csv"), np.array(LOG_TIMES), delimiter=",")
                print(f"Log data saved to: {SAVE_DIR}")
                break
            #if step > 2000:
            #    print(f"Run {i+1}: Timeout")
            #    break

        # Save logs for this run
        
        

def simulate():
    from mujoco import viewer
    
    global goal_xy, LOG_STATES, LOG_ACTIONS, LOG_TIMES, U_global

    goal_reached = True
    n_runs = 100
    for i in range(n_runs):
        goal_xy = np.array([0.01 * (i + 200), 0.0])
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
                #print(data.qpos[0], data.qpos[1], data.qpos[2])
                log_data(data, data.ctrl)
                
                dist_to_goal = np.linalg.norm(data.qpos[:2] - goal_xy)
                if (data.qpos[2] < 0.15):
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

        #if goal_reached:
        #    save_logs()     



def simulate1():
    from mujoco import viewer
    
    goal_reached = False
    
    with viewer.launch_passive(model, data) as v:
        print("Simulation running. Close the viewer window to stop.")
        while v.is_running():
            mppi_update(model, data)
            mujoco.mj_step(model, data)
            v.sync()
            print(data.qpos[0], data.qpos[1], data.qpos[2])
            log_data(data, data.ctrl)
            global goal_xy
            
            dist_to_goal = np.linalg.norm(data.qpos[:2] - goal_xy)
            if (data.qpos[2] < 0.15):
                print(f"Robot too low")
                goal_reached = False
                break
            if dist_to_goal < GOAL_TOLERANCE:
                goal_reached = True
                print("Goal reached!")
                break
            if data.qpos[0] >= goal_xy[0]:
                print(data.qpos[0], data.qpos[0])
                goal_reached = True
                print("Goal Line (X) reached!")
                break  

    if goal_reached:
        save_logs()


# Main
if __name__ == "__main__":
    #simulate_headless_n_runs(n_runs=100)
    simulate()