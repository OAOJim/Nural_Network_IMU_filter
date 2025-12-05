import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os

# CONFIG
DT = 0.01
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GLOBAL CONSTANTS
MAG_REF_WORLD = np.array([20.0, 0.0, -40.0])
G_VECTOR = np.array([0.0, 0.0, 9.81])       

def q_to_euler_numpy(q):
    """Converts a batch of quaternions to Euler angles (Roll, Pitch, Yaw)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    sinr_cosp = 2 * (w * x + y * z); cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    sinp = 2 * (w * y - z * x)
    sinp = np.where(np.abs(sinp) >= 1, np.sign(sinp) * (1 - 1e-6), sinp)
    pitch = np.arcsin(sinp)
    siny_cosp = 2 * (w * z + x * y); cosy_cosp = 1 - 2 * (y * y + z * z)
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

## 2 Model
# Model 1 Position/Velocity (6x6 output)
class KalmanNetPV(nn.Module):
    def __init__(self):
        super(KalmanNetPV, self).__init__()
        self.state_dim_pv = 6
        self.obs_dim = 6
        self.hidden_dim = 64
        self.nn_input_dim = self.obs_dim + 7 
        self.norm = nn.LayerNorm(self.nn_input_dim)
        self.gru = nn.GRUCell(input_size=self.nn_input_dim, hidden_size=self.hidden_dim) 
        self.fc_k_gain = nn.Linear(self.hidden_dim, self.state_dim_pv * self.obs_dim) 

# Model 2: Orientation Corrector (4x6 output)
class KalmanNetQ(nn.Module):
    def __init__(self):
        super(KalmanNetQ, self).__init__()
        self.state_dim_q = 4
        self.obs_dim = 6
        self.hidden_dim = 64
        self.nn_input_dim = self.obs_dim + 7 
        self.norm = nn.LayerNorm(self.nn_input_dim)
        self.gru = nn.GRUCell(input_size=self.nn_input_dim, hidden_size=self.hidden_dim) 
        self.fc_k_gain = nn.Linear(self.hidden_dim, self.state_dim_q * self.obs_dim) 

class HybridKNet:
    def __init__(self, knet_pv: KalmanNetPV, knet_q: KalmanNetQ):
        self.knet_pv = knet_pv
        self.knet_q = knet_q
        self.state_dim_full = 10
        self.obs_dim = 6
        self.hidden_dim = 64
        
        self.g_vec = torch.tensor(G_VECTOR, dtype=torch.float32, device=device)
        self.mag_ref = torch.tensor(MAG_REF_WORLD, dtype=torch.float32, device=device)

    def run_sequence(self, inputs, initial_state):
        batch_size, seq_len, _ = inputs.shape
        g_vec_batch = self.g_vec.unsqueeze(0).expand(batch_size, -1)
        mag_ref_batch = self.mag_ref.unsqueeze(0).expand(batch_size, -1)
        
        # Initialize state and GRU hidden states (one for each NN)
        x_curr = initial_state.clone()
        h_gru_pv = torch.zeros(batch_size, self.hidden_dim, device=device)
        h_gru_q = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        outputs = []

        for t in range(seq_len):
            # 1. Sensor Inputs
            accel_meas = inputs[:, t, 0:3]
            gyro_meas = inputs[:, t, 3:6]
            mag_meas = inputs[:, t, 6:9]

            p, v, q = x_curr[:, 0:3], x_curr[:, 3:6], x_curr[:, 6:10]
            
            # 2. Prediction 
            
            # 2a. Orientation Integration (Dead Reckoning)
            q_for_rotation = q # Use estimated quaternion for inference
            half_ang = gyro_meas * DT * 0.5
            ones = torch.ones(batch_size, 1, device=device)
            delta_q = torch.cat((ones, half_ang), dim=1)
            delta_q = delta_q / torch.norm(delta_q, dim=1, keepdim=True)
            q_pred = q_multiply(q, delta_q)
            q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)

            # 2b. Position/Velocity Integration
            accel_world = rotate_vec(q_for_rotation, accel_meas) - g_vec_batch 
            v_pred = v + accel_world * DT
            p_pred = p + v * DT + 0.5 * accel_world * DT**2
            
            x_pred_pv = torch.cat((p_pred, v_pred), dim=1)
            x_pred_full = torch.cat((x_pred_pv, q_pred), dim=1)

            # 3. innovation 
            g_body_pred = rotate_vec_inverse(q_pred, g_vec_batch)
            mag_body_pred = rotate_vec_inverse(q_pred, mag_ref_batch)
            y_meas = torch.cat((mag_meas, accel_meas), dim=1)
            y_pred = torch.cat((mag_body_pred, g_body_pred), dim=1)
            innovation = y_meas - y_pred 
            
            # 4. correction
            dynamic_state = x_pred_full[:, 3:10] # V(3) + Q(4)
            gru_in = torch.cat((innovation, dynamic_state), dim=1)
            
            gru_in_norm = self.knet_pv.norm(gru_in) 
            
            # 4a. PV Correction (Model 1)
            h_gru_pv = self.knet_pv.gru(gru_in_norm, h_gru_pv)
            K_pv_flat = self.knet_pv.fc_k_gain(h_gru_pv)
            K_pv = K_pv_flat.reshape(batch_size, self.knet_pv.state_dim_pv, self.obs_dim) # 6x6
            correction_pv = torch.bmm(K_pv, innovation.unsqueeze(2)).squeeze(2)

            # 4b. Q Correction (Model 2)
            h_gru_q = self.knet_q.gru(gru_in_norm, h_gru_q)
            K_q_flat = self.knet_q.fc_k_gain(h_gru_q)
            K_q = K_q_flat.reshape(batch_size, self.knet_q.state_dim_q, self.obs_dim) # 4x6
            correction_q = torch.bmm(K_q, innovation.unsqueeze(2)).squeeze(2)

            # Combine and finalize state
            p_new = x_pred_pv[:, 0:3] + correction_pv[:, 0:3]
            v_new = x_pred_pv[:, 3:6] + correction_pv[:, 3:6]
            q_new = q_pred + correction_q
            
            # Re-normalize Quaternion
            q_final = q_new / torch.norm(q_new, dim=1, keepdim=True)
            
            x_curr = torch.cat((p_new, v_new, q_final), dim=1)
            outputs.append(x_curr)

        return torch.stack(outputs, dim=1)


# ==========================================
# 4. CLASSIC EKF (For comparison)
# ==========================================
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
        w1, x1, y1, z1 = q; w2, x2, y2, z2 = dq
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
        self.P = self.P + self.Q 
        return x_pred

    def correct(self, x_pred, sensors):
        H = self.numerical_jacobian_h(x_pred)
        y_pred = self.predict_sensors(x_pred)
        innovation = sensors - y_pred
        S = H @ self.P @ H.T + self.R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
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
        # Jacobian only for the Quaternions (indices 6:10)
        for i in range(6, 10): 
            x_plus = x.copy(); x_plus[i] += epsilon
            h_plus = self.predict_sensors(x_plus)
            H[:, i] = (h_plus - base_h) / epsilon
        return H

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def load_knet_models():
    # Load Position NN
    knet_pv = KalmanNetPV().to(device)
    pv_path = "kalmannet_pruned.pth"
    if os.path.exists(pv_path):
        knet_pv.load_state_dict(torch.load(pv_path, map_location=device))
        print(f"Loaded KNet PV model from {pv_path}.")
    else:
        print(f"Warning: PV Model not found ({pv_path}). Running untrained.")
    knet_pv.eval()
    
    # Load Orientation NN
    knet_q = KalmanNetQ().to(device)
    q_path = "kalmannet_model_q_euler.pth"
    if os.path.exists(q_path):
        knet_q.load_state_dict(torch.load(q_path, map_location=device))
        print(f"Loaded KNet Q model from {q_path}.")
    else:
        print(f"Warning: Q Model not found ({q_path}). Running untrained.")
    knet_q.eval()
    
    return knet_pv, knet_q

def compare_all():
    # Load Data
    try:
        inputs_raw = pd.read_csv('data_no_bias/test_inputs.csv', header=None).values
        targets_raw = pd.read_csv('data_no_bias/test_targets.csv', header=None).values
        print("Loaded dataset files.")
    except:
        print("Datasets not found. Generating dummy data...")
        # Since generating dummy data is non-trivial for IMU, we skip that for now and expect the file structure.
        print("Error: Required data files not found. Exiting.")
        return

    # Load Models
    knet_pv, knet_q = load_knet_models()
    hybrid_knet = HybridKNet(knet_pv, knet_q)
    
    # EKF TUNING 
    Q_cov = np.eye(10) * 1e-3
    Q_cov[6:10, 6:10] *= 1e-5 # Trust gyro for orientation process
    R_cov = np.eye(6) * 1.0 # Trust sensors moderately
    # ----------------------------------------

    SEGMENT_LEN = 200 
    num_segments = (len(inputs_raw) - 1) // SEGMENT_LEN
    
    # Metrics Lists
    rmse_pos_base, rmse_pos_ekf, rmse_pos_knet = [], [], []
    rmse_q_base, rmse_q_ekf, rmse_q_knet = [], [], []

    viz_traj_gt, viz_traj_ekf, viz_traj_knet = [], [], []
    viz_euler_gt, viz_euler_ekf, viz_euler_knet = [], [], []

    print(f"Running comparison on {num_segments} segments...")
    
    with torch.no_grad():
        for i in range(num_segments):
            progress = (i + 1) / num_segments
            print(f"Progress: {progress * 100:.2f}% ({i+1}/{num_segments} segments)", end='\r')

            start = i * SEGMENT_LEN
            end = start + SEGMENT_LEN
            start_truth = targets_raw[start]
            
            # KNet Prep
            seg_in_torch = torch.tensor(inputs_raw[start:end], dtype=torch.float32).unsqueeze(0).to(device)
            state_in_torch = torch.tensor(start_truth, dtype=torch.float32).unsqueeze(0).to(device)
            
            # Run Hybrid KNet
            knet_out = hybrid_knet.run_sequence(seg_in_torch, state_in_torch)
            knet_full = knet_out.squeeze(0).cpu().numpy()
            
            # Run EKF and Base (Dead Reckoning)
            ekf_path = []
            base_path = []
            x_curr_ekf = start_truth.copy()
            x_curr_base = start_truth.copy()
            ekf = ClassicEKF(R_cov, Q_cov)
            
            for t in range(start, end):
                gyro = inputs_raw[t, 3:6]; accel = inputs_raw[t, 0:3]; mag = inputs_raw[t, 6:9]
                sensors = np.concatenate([mag, accel])
                
                # EKF
                x_pred_ekf = ekf.predict(x_curr_ekf, gyro, accel)
                x_curr_ekf = ekf.correct(x_pred_ekf, sensors)
                ekf_path.append(x_curr_ekf)

                # Baseline (Dead Reckoning: Just Prediction)
                x_curr_base = ekf.predict(x_curr_base, gyro, accel) # Reuse EKF predict for DR
                base_path.append(x_curr_base)
            
            ekf_full = np.array(ekf_path)
            base_full = np.array(base_path)
            truth_full = targets_raw[start+1 : end+1]

            # --- Metrics Calculation ---
            
            # Position (3D distance error)
            pos_err_base = np.linalg.norm(base_full[:,0:3] - truth_full[:,0:3], axis=1)
            pos_err_ekf  = np.linalg.norm(ekf_full[:,0:3]  - truth_full[:,0:3], axis=1)
            pos_err_knet = np.linalg.norm(knet_full[:,0:3] - truth_full[:,0:3], axis=1)
            
            # Orientation (Quaternion error, simple magnitude difference)
            q_err_base = np.linalg.norm(base_full[:,6:10] - truth_full[:,6:10], axis=1)
            q_err_ekf  = np.linalg.norm(ekf_full[:,6:10]  - truth_full[:,6:10], axis=1)
            q_err_knet = np.linalg.norm(knet_full[:,6:10] - truth_full[:,6:10], axis=1)
            
            rmse_pos_base.append(np.mean(pos_err_base))
            rmse_pos_ekf.append(np.mean(pos_err_ekf))
            rmse_pos_knet.append(np.mean(pos_err_knet))

            rmse_q_base.append(np.mean(q_err_base))
            rmse_q_ekf.append(np.mean(q_err_ekf))
            rmse_q_knet.append(np.mean(q_err_knet))
            
            # Visualization data (last segment)
            if i == num_segments - 1:
                viz_traj_gt = truth_full[:, 0:3]
                viz_traj_ekf = ekf_full[:, 0:3]
                viz_traj_knet = knet_full[:, 0:3]
                viz_euler_gt = q_to_euler_numpy(truth_full[:, 6:10])
                viz_euler_ekf = q_to_euler_numpy(ekf_full[:, 6:10])
                viz_euler_knet = q_to_euler_numpy(knet_full[:, 6:10])
    
    print("Comparison complete. Calculating final statistics...")

    # Final Stats
    print("\n--- FINAL RESULTS (Averaged over segments) ---")
    print(f"{'Metric':<20} | {'Base':<10} | {'EKF':<10} | {'KNet (Hybrid)':<15}")
    print("-" * 60)
    print(f"{'Position RMS Error (m)':<20} | {np.mean(rmse_pos_base):.3f}        | {np.mean(rmse_pos_ekf):.3f}        | {np.mean(rmse_pos_knet):.3f}")
    print(f"{'Quat RMS Error (mag)':<20} | {np.mean(rmse_q_base):.6f}      | {np.mean(rmse_q_ekf):.6f}      | {np.mean(rmse_q_knet):.6f}")

    # Run Efficiency Test Last
    # We test the inference time for the hybrid KNet
    test_efficiency(hybrid_knet, device)

    plot_results(viz_traj_gt, viz_traj_ekf, viz_traj_knet, viz_euler_gt, viz_euler_ekf, viz_euler_knet)
    print("\nPlots saved.")

# EFFICIENCY TESTING 
def test_efficiency(hybrid_knet, device):
    print("\n--- EFFICIENCY ANALYSIS (Hybrid KNet) ---")
    batch_size = 1; seq_len = 1000; num_runs = 50
    input_dim = 9
    dummy_input = torch.randn(batch_size, seq_len, input_dim).to(device)
    dummy_state = torch.zeros(batch_size, 10, device=device); dummy_state[:, 6] = 1.0
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = hybrid_knet.run_sequence(dummy_input, dummy_state)
            
    # Timing
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = hybrid_knet.run_sequence(dummy_input, dummy_state)
            
    total_time = time.time() - start_time
    avg_time_per_seq = total_time / num_runs
    avg_time_per_step = avg_time_per_seq / seq_len
    print(f"  Average Latency per Step: {avg_time_per_step*1000*1000:.4f} microseconds")
    print("---------------------------------------")

# PLOT
def plot_results(traj_gt, traj_ekf, traj_knet, euler_gt, euler_ekf, euler_knet):
    # 1. Trajectory
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(traj_gt[:,0]-traj_gt[0,0], traj_gt[:,1]-traj_gt[0,1], traj_gt[:,2]-traj_gt[0,2], 'k-', label='Ground Truth')
    ax.plot(traj_ekf[:,0]-traj_ekf[0,0], traj_ekf[:,1]-traj_ekf[0,1], traj_ekf[:,2]-traj_ekf[0,2], 'g--', label='EKF')
    ax.plot(traj_knet[:,0]-traj_knet[0,0], traj_knet[:,1]-traj_knet[0,1], traj_knet[:,2]-traj_knet[0,2], 'b-.', label='KNet (Hybrid)')
    ax.legend()
    ax.set_title("3D Trajectory Reconstruction (Relative to Start)")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    plt.savefig('trajectory_3d.png')
    plt.close()

    # 2. Euler
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    labels = ['Roll', 'Pitch', 'Yaw']
    
    euler_gt_deg = np.degrees(euler_gt)
    euler_ekf_deg = np.degrees(euler_ekf)
    euler_knet_deg = np.degrees(euler_knet)
    # Unwarp
    for i in range(3):
        euler_gt_deg[:, i] = np.unwrap(euler_gt_deg[:, i], discont=360.0)
        euler_ekf_deg[:, i] = np.unwrap(euler_ekf_deg[:, i], discont=360.0)
        euler_knet_deg[:, i] = np.unwrap(euler_knet_deg[:, i], discont=360.0)

    for i in range(3):
        axs[i].plot(euler_gt_deg[:, i], 'k-', label='Truth')
        axs[i].plot(euler_ekf_deg[:, i], 'g--', label='EKF', alpha=0.7)
        axs[i].plot(euler_knet_deg[:, i], 'b-.', label='KNet (Hybrid)', alpha=0.7)
        axs[i].set_ylabel(f"{labels[i]} (deg)")
        axs[i].grid(True, alpha=0.3)
    
    axs[0].set_title("Orientation Estimation (Euler Angles - Unwrapped)")
    axs[0].legend()
    plt.xlabel("Time Steps")
    plt.savefig('orientation_euler.png')
    plt.close()

if __name__ == "__main__":
    compare_all()