import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Config 
DT = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constant
MAG_REF_WORLD = np.array([20.0, 0.0, -40.0]) # The ground truth magnetic field vector
G_VECTOR = np.array([0.0, 0.0, 9.81])       # Gravity in World Frame
# 1. math functions
def q_to_euler_numpy(q):
    """
    Convert batch of quaternions to Euler angles (Roll, Pitch, Yaw).
    q: (N, 4) [w, x, y, z]
    Returns: (N, 3) [roll, pitch, yaw] in Radians
    """
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    # Roll
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch
    sinp = 2 * (w * y - z * x)
    # Clamp argument to arcsin to avoid NaNs due to floating point errors
    sinp = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (1 - 1e-6), sinp)
    pitch = np.arcsin(sinp)

    # Yaw
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack((roll, pitch, yaw), axis=1)

def q_to_rot_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2*(y**2 + z**2); r01 = 2*(x*y - z*w);      r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w);        r11 = 1 - 2*(x**2 + z**2); r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w);        r21 = 2*(y*z + x*w);       r22 = 1 - 2*(x**2 + y**2)
    row0 = torch.stack((r00, r01, r02), dim=1)
    row1 = torch.stack((r10, r11, r12), dim=1)
    row2 = torch.stack((r20, r21, r22), dim=1)
    return torch.stack((row0, row1, row2), dim=1)

def q_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=1)

def rotate_vec(q, v):
    R = q_to_rot_matrix(q) 
    v_expanded = v.unsqueeze(-1)
    v_rotated = torch.bmm(R, v_expanded)
    return v_rotated.squeeze(-1)

def rotate_vec_inverse(q, v):
    R = q_to_rot_matrix(q)
    v_expanded = v.unsqueeze(-1)
    v_rotated = torch.bmm(R.transpose(1, 2), v_expanded)
    return v_rotated.squeeze(-1)

# NumPy Helpers
def q_to_rot_matrix_numpy(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),       1 - 2*(x**2 + y**2)]
    ])

def rotate_vec_inverse_numpy(q, v):
    R = q_to_rot_matrix_numpy(q)
    return R.T @ v

def rotate_vec_numpy(q, v):
    R = q_to_rot_matrix_numpy(q)
    return R @ v

# 2. Model
class KalmanNet(nn.Module):
    def __init__(self):
        super(KalmanNet, self).__init__()
        self.state_dim_full = 10 # Pos(3), Vel(3), Quat(4)
        self.state_dim_pv = 6    # Pos(3), Vel(3) - States corrected by NN
        self.obs_dim = 6         # Mag(3) + Accel(3) - Inputs to NN
        self.hidden_dim = 64
        
        # NN Input: Innovation (6) + Velocity (3) + Quaternion (4) = 13 inputs
        self.nn_input_dim = self.obs_dim + 7 
        
        self.norm = nn.LayerNorm(self.nn_input_dim)
        self.gru = nn.GRUCell(input_size=self.nn_input_dim, hidden_size=self.hidden_dim) 
        
        # Output layer 
        self.fc_k_gain = nn.Linear(self.hidden_dim, self.state_dim_pv * self.obs_dim) 

        # Fixed constants
        self.g_vec = torch.tensor(G_VECTOR, dtype=torch.float32, device=device)
        self.mag_ref = torch.tensor(MAG_REF_WORLD, dtype=torch.float32, device=device)

    def forward(self, inputs, initial_state_tuple=None, q_gt_seq=None): # ADDED q_gt_seq
        batch_size, seq_len, _ = inputs.shape
        g_vec_batch = self.g_vec.unsqueeze(0).expand(batch_size, -1)
        mag_ref_batch = self.mag_ref.unsqueeze(0).expand(batch_size, -1)

        if initial_state_tuple is None:
            x_curr = torch.zeros(batch_size, self.state_dim_full, device=device)
            x_curr[:, 6] = 1.0 
            h_gru = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            x_curr, h_gru = initial_state_tuple

        outputs = []

        for t in range(seq_len):
            accel_meas = inputs[:, t, 0:3]
            gyro_meas = inputs[:, t, 3:6]
            mag_meas = inputs[:, t, 6:9]

            p, v, q = x_curr[:, 0:3], x_curr[:, 3:6], x_curr[:, 6:10]

            # --- Prediction ---
            if q_gt_seq is not None:
                q_for_rotation = q_gt_seq[:, t, :] 
            else:
                q_for_rotation = q 
            
            # 2a. Orientation (Dead Reckoning)
            half_ang = gyro_meas * DT * 0.5
            ones = torch.ones(batch_size, 1, device=device)
            delta_q = torch.cat((ones, half_ang), dim=1)
            delta_q = delta_q / torch.norm(delta_q, dim=1, keepdim=True)
            q_pred = q_multiply(q, delta_q)
            q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)

            # 2b. Position/Velocity
            accel_world = rotate_vec(q_for_rotation, accel_meas) - g_vec_batch 
            v_pred = v + accel_world * DT
            p_pred = p + v * DT + 0.5 * accel_world * DT**2
            
            x_pred_pv = torch.cat((p_pred, v_pred), dim=1)
            x_pred_full = torch.cat((x_pred_pv, q_pred), dim=1)
            # --- Observation and Innovation ---
            g_body_pred = rotate_vec_inverse(q_pred, g_vec_batch)
            mag_body_pred = rotate_vec_inverse(q_pred, mag_ref_batch)
            y_meas = torch.cat((mag_meas, accel_meas), dim=1)
            y_pred = torch.cat((mag_body_pred, g_body_pred), dim=1)
            innovation = y_meas - y_pred 
            
            # --- Correction ---
            dynamic_state = x_pred_full[:, 3:10] # V(3) + Q(4)
            gru_in = torch.cat((innovation, dynamic_state), dim=1)
            gru_in = self.norm(gru_in)
            h_gru = self.gru(gru_in, h_gru)
            
            k_flat = self.fc_k_gain(h_gru)
            K = k_flat.reshape(batch_size, self.state_dim_pv, self.obs_dim) 
            
            innovation_expanded = innovation.unsqueeze(2) 
            correction_pv = torch.bmm(K, innovation_expanded).squeeze(2)
            # Build the new full state
            p_new = x_pred_pv[:, 0:3] + correction_pv[:, 0:3]
            v_new = x_pred_pv[:, 3:6] + correction_pv[:, 3:6]
            # Orientation: Keep the predicted quaternion
            q_final = q_pred 
            
            x_curr = torch.cat((p_new, v_new, q_final), dim=1)
            outputs.append(x_curr)

        return torch.stack(outputs, dim=1)

# 3. CLASSIC EKF
class ClassicEKF:
    def __init__(self, R_cov, Q_cov):
        self.R = R_cov
        self.Q = Q_cov
        self.P = np.eye(10) * 0.1
        self.g_vec = G_VECTOR
        self.mag_ref = MAG_REF_WORLD

    def predict(self, x, gyro_meas, accel_meas):
        p, v, q = x[0:3], x[3:6], x[6:10]
        
        half_ang = gyro_meas * DT * 0.5
        dq = np.array([1.0, half_ang[0], half_ang[1], half_ang[2]])
        dq = dq / np.linalg.norm(dq)
        
        w1, x1, y1, z1 = q
        w2, x2, y2, z2 = dq
        q_pred = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        q_pred = q_pred / np.linalg.norm(q_pred)
        
        accel_world = rotate_vec_numpy(q, accel_meas) - self.g_vec
        v_pred = v + accel_world * DT
        p_pred = p + v * DT + 0.5 * accel_world * DT**2
        
        x_pred = np.concatenate([p_pred, v_pred, q_pred])
        # Only inflate covariance for Position/Velocity to focus on that correction
        self.P[0:6, 0:6] = self.P[0:6, 0:6] + self.Q[0:6, 0:6]
        return x_pred

    def correct(self, x_pred, sensors):
        H = self.numerical_jacobian_h(x_pred)
        y_pred = self.predict_sensors(x_pred)
        innovation = sensors - y_pred
        
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            print("Warning: S matrix is singular. Using zero gain.")
            K = np.zeros((10, 6))

        x_new = x_pred + K @ innovation
        x_new[6:10] = x_new[6:10] / np.linalg.norm(x_new[6:10])
        
        I = np.eye(10)
        self.P = (I - K @ H) @ self.P
        return x_new

    def predict_sensors(self, x):
        q = x[6:10]
        g_body = rotate_vec_inverse_numpy(q, self.g_vec)
        mag_body = rotate_vec_inverse_numpy(q, self.mag_ref)
        return np.concatenate([mag_body, g_body])

    def numerical_jacobian_h(self, x):
        epsilon = 1e-4
        H = np.zeros((6, 10))
        base_h = self.predict_sensors(x)
        # Jacobian only for the Quaternions
        for i in range(6, 10): 
            x_plus = x.copy()
            x_plus[i] += epsilon
            h_plus = self.predict_sensors(x_plus)
            H[:, i] = (h_plus - base_h) / epsilon
        return H

# PLOT
def generate_dummy_data(length=500):
    print("Generating dummy data for demonstration...")
    t = np.linspace(0, 20, length)
    p_x = 5 * np.cos(t); p_y = 5 * np.sin(t); p_z = t * 0.1
    v_x = -5 * np.sin(t); v_y = 5 * np.cos(t); v_z = np.ones_like(t) * 0.1
    
    roll = np.zeros_like(t); pitch = np.zeros_like(t); yaw = t
    cy = np.cos(yaw * 0.5); sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5); sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5); sr = np.sin(roll * 0.5)
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    
    targets = np.column_stack([p_x, p_y, p_z, v_x, v_y, v_z, qw, qx, qy, qz])
    accel_noise = np.random.normal(0, 0.1, (length, 3))
    gyro_noise = np.random.normal(0, 0.01, (length, 3))
    mag_noise = np.random.normal(0, 0.1, (length, 3))
    
    inputs = np.zeros((length, 9))
    inputs[:, 3:6] = np.column_stack([np.zeros_like(t), np.zeros_like(t), np.ones_like(t)]) + gyro_noise
    inputs[:, 0:3] = accel_noise
    inputs[:, 6:9] = mag_noise
    
    return inputs, targets

def test_efficiency(model, device):
    print("\n--- EFFICIENCY ANALYSIS (KalmanNet) ---")
    model.eval()
    batch_size = 1
    seq_len = 1000 
    num_runs = 50
    input_dim = 9
    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)
    print(f"Warming up ({device})...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)
    print(f"Running {num_runs} iterations (Seq Len: {seq_len})...")
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            # In inference mode, q_gt_seq is None (using estimated q)
            _ = model(dummy_input, q_gt_seq=None)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_seq = total_time / num_runs
    avg_time_per_step = avg_time_per_seq / seq_len
    print(f"  Average Latency per Step: {avg_time_per_step*1000*1000:.4f} microseconds")
    print("---------------------------------------")

def plot_results(traj_gt, traj_ekf, traj_knet, euler_gt, euler_ekf, euler_knet):
    # 1. Trajectory
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_gt[:,0]-traj_gt[0,0], traj_gt[:,1]-traj_gt[0,1], traj_gt[:,2]-traj_gt[0,2], 'k-', label='Ground Truth')
    ax.plot(traj_ekf[:,0]-traj_ekf[0,0], traj_ekf[:,1]-traj_ekf[0,1], traj_ekf[:,2]-traj_ekf[0,2], 'g--', label='EKF')
    ax.plot(traj_knet[:,0]-traj_knet[0,0], traj_knet[:,1]-traj_knet[0,1], traj_knet[:,2]-traj_knet[0,2], 'b-.', label='KalmanNet')
    ax.legend()
    ax.set_title("3D Trajectory Reconstruction (Relative to Start)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    plt.savefig('trajectory_3d_position.png')
    plt.close()

    # 2. Euler
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    
    euler_gt_deg = np.degrees(euler_gt)
    euler_ekf_deg = np.degrees(euler_ekf)
    euler_knet_deg = np.degrees(euler_knet)
    
    # UNWRAP the ground truth and estimated angles to fix the visualization jumps
    for i in range(3):
        euler_gt_deg[:, i] = np.unwrap(euler_gt_deg[:, i], discont=360.0)
        euler_ekf_deg[:, i] = np.unwrap(euler_ekf_deg[:, i], discont=360.0)
        euler_knet_deg[:, i] = np.unwrap(euler_knet_deg[:, i], discont=360.0)


    for i in range(3):
        axs[i].plot(euler_gt_deg[:, i], 'k-', label='Truth')
        axs[i].plot(euler_ekf_deg[:, i], 'g--', label='EKF', alpha=0.7)
        axs[i].plot(euler_knet_deg[:, i], 'b-.', label='KalmanNet', alpha=0.7)
        axs[i].set_ylabel(f"{labels[i]} (deg)")
        axs[i].grid(True, alpha=0.3)
    
    axs[0].set_title("Orientation Estimation (Euler Angles - Unwrapped)")
    axs[0].legend()
    plt.xlabel("Time Steps")
    plt.savefig('orientation_euler.png')
    plt.close()

# main 
def compare_all():
    # Load Data
    try:
        inputs_raw = pd.read_csv('data_no_bias/test_inputs.csv', header=None).values
        targets_raw = pd.read_csv('data_no_bias/test_targets.csv', header=None).values
        print("Loaded dataset files.")
    except:
        print("Datasets not found. Generating dummy data...")
        inputs_raw, targets_raw = generate_dummy_data()

    # Load Model
    kn_model = KalmanNet().to(device)
    try:
        kn_model.load_state_dict(torch.load("kalmannet_pruned.pth", map_location=device)) # use pruned data
        kn_model.eval()
        print("Loaded KalmanNet model.")
    except:
        print("Warning: Model file not found. Running with untrained KNet.")
        kn_model.eval()
    
    # EKF TUNING WWWW
    Q_cov = np.eye(10) * 1e-3
    Q_cov[6:10, 6:10] *= 1e-5 # Very low Q for orientation
    
    R_cov = np.eye(6)
    R_cov[0:3, 0:3] *= 1.0    # Mag R (Yaw correction)
    R_cov[3:6, 3:6] *= 0.5    # Accel R (Tilt correction) 
    # ----------------------------------------

    SEGMENT_LEN = 200 
    num_segments = (len(inputs_raw) - 1) // SEGMENT_LEN
    
    rmse_pos_base, rmse_pos_ekf, rmse_pos_knet = [], [], []
    rmse_tilt_base, rmse_tilt_ekf, rmse_tilt_knet = [], [], []
    rmse_yaw_base, rmse_yaw_ekf, rmse_yaw_knet = [], [], []

    viz_traj_gt, viz_traj_ekf, viz_traj_knet = [], [], []
    viz_euler_gt, viz_euler_ekf, viz_euler_knet = [], [], []

    print(f"Running comparison on {num_segments} segments...")
    
    with torch.no_grad():
        for i in range(num_segments):
            # --- Progress Bar Update ---
            progress = (i + 1) / num_segments
            print(f"Progress: {progress * 100:.2f}% ({i+1}/{num_segments} segments)", end='\r')
            # ---------------------------

            start = i * SEGMENT_LEN
            end = start + SEGMENT_LEN
            start_truth = targets_raw[start]
            
            # KNet Prep
            seg_in_torch = torch.tensor(inputs_raw[start:end], dtype=torch.float32).unsqueeze(0).to(device)
            state_in_torch = torch.tensor(start_truth, dtype=torch.float32).unsqueeze(0).to(device)
            h_in_torch = torch.zeros(1, 64, device=device)
            
            # Run Models
            # q_gt_seq is left as None here, so KNet uses its estimated quaternion (inference mode)
            knet_out = kn_model(seg_in_torch, initial_state_tuple=(state_in_torch, h_in_torch), q_gt_seq=None)
            knet_full = knet_out.squeeze(0).cpu().numpy()
            
            ekf_path = []
            x_curr = start_truth.copy()
            ekf = ClassicEKF(R_cov, Q_cov)
            
            base_path = []
            x_base = start_truth.copy()
            # Reuse EKF class for physics predict step in baseline
            ekf_base_helper = ClassicEKF(R_cov, Q_cov) 
            
            for t in range(start, end):
                gyro = inputs_raw[t, 3:6]
                accel = inputs_raw[t, 0:3]
                mag = inputs_raw[t, 6:9]
                
                # EKF
                x_pred = ekf.predict(x_curr, gyro, accel)
                sensors = np.concatenate([mag, accel])
                x_curr = ekf.correct(x_pred, sensors)
                ekf_path.append(x_curr)

                # Baseline (Dead Reckoning)
                x_base = ekf_base_helper.predict(x_base, gyro, accel)
                base_path.append(x_base)
            
            ekf_full = np.array(ekf_path)
            base_full = np.array(base_path)
            truth_full = targets_raw[start+1 : end+1]

            # Metrics
            # Position
            pos_err_base = np.linalg.norm(base_full[:,0:3] - truth_full[:,0:3], axis=1)
            pos_err_ekf  = np.linalg.norm(ekf_full[:,0:3]  - truth_full[:,0:3], axis=1)
            pos_err_knet = np.linalg.norm(knet_full[:,0:3] - truth_full[:,0:3], axis=1)
            
            # Orientation
            euler_gt = q_to_euler_numpy(truth_full[:, 6:10])
            euler_base = q_to_euler_numpy(base_full[:, 6:10])
            euler_ekf = q_to_euler_numpy(ekf_full[:, 6:10])
            euler_knet = q_to_euler_numpy(knet_full[:, 6:10])
            
            for j in range(3):
                euler_gt[:, j] = np.unwrap(euler_gt[:, j], discont=2*np.pi)
                euler_base[:, j] = np.unwrap(euler_base[:, j], discont=2*np.pi)
                euler_ekf[:, j] = np.unwrap(euler_ekf[:, j], discont=2*np.pi)
                euler_knet[:, j] = np.unwrap(euler_knet[:, j], discont=2*np.pi)


            def ang_diff(a, b):
                # Simple difference after unwrap, since jumps are removed
                return np.abs(a - b)

            tilt_diff_base = np.mean(ang_diff(euler_base[:, :2], euler_gt[:, :2]), axis=1)
            yaw_diff_base = ang_diff(euler_base[:, 2], euler_gt[:, 2])

            tilt_diff_ekf = np.mean(ang_diff(euler_ekf[:, :2], euler_gt[:, :2]), axis=1)
            yaw_diff_ekf = ang_diff(euler_ekf[:, 2], euler_gt[:, 2])

            tilt_diff_knet = np.mean(ang_diff(euler_knet[:, :2], euler_gt[:, :2]), axis=1)
            yaw_diff_knet = ang_diff(euler_knet[:, 2], euler_gt[:, 2])

            rmse_pos_base.append(np.mean(pos_err_base))
            rmse_pos_ekf.append(np.mean(pos_err_ekf))
            rmse_pos_knet.append(np.mean(pos_err_knet))

            rmse_tilt_base.append(np.mean(np.degrees(tilt_diff_base)))
            rmse_tilt_ekf.append(np.mean(np.degrees(tilt_diff_ekf)))
            rmse_tilt_knet.append(np.mean(np.degrees(tilt_diff_knet)))

            rmse_yaw_base.append(np.mean(np.degrees(yaw_diff_base)))
            rmse_yaw_ekf.append(np.mean(np.degrees(yaw_diff_ekf)))
            rmse_yaw_knet.append(np.mean(np.degrees(yaw_diff_knet)))
            
            if i == num_segments - 1:
                viz_traj_gt = truth_full[:, 0:3]
                viz_traj_ekf = ekf_full[:, 0:3]
                viz_traj_knet = knet_full[:, 0:3]
                viz_euler_gt = euler_gt
                viz_euler_ekf = euler_ekf
                viz_euler_knet = euler_knet
    
    print("Comparison complete. Calculating final statistics...")

    # Final Stats
    print("\n--- FINAL RESULTS (Averaged over segments) ---")
    print(f"{'Metric':<15} | {'Base':<10} | {'EKF':<10} | {'KNet':<10}")
    print("-" * 55)
    print(f"{'Position (m)':<15} | {np.mean(rmse_pos_base):.3f}        | {np.mean(rmse_pos_ekf):.3f}        | {np.mean(rmse_pos_knet):.3f}")
    print(f"{'Tilt (deg)':<15} | {np.mean(rmse_tilt_base):.3f}        | {np.mean(rmse_tilt_ekf):.3f}        | {np.mean(rmse_tilt_knet):.3f}")
    print(f"{'Yaw (deg)':<15} | {np.mean(rmse_yaw_base):.3f}        | {np.mean(rmse_yaw_ekf):.3f}        | {np.mean(rmse_yaw_knet):.3f}")

    # Run Efficiency Test Last
    test_efficiency(kn_model, device)

    plot_results(viz_traj_gt, viz_traj_ekf, viz_traj_knet, viz_euler_gt, viz_euler_ekf, viz_euler_knet)
    print("\nPlots saved.")

if __name__ == "__main__":
    compare_all()