import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import shutil
import torch.nn.functional as F

# Config
DT = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "results_comparison"
print(f"Running on: {DEVICE}")

# output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)
print(f"Figures will be saved to: {OUTPUT_DIR}/")

# Physics Constants
G_VECTOR = np.array([0.0, 0.0, 9.81])
MAG_REF_WORLD = np.array([20.0, 0.0, -40.0])

# math Functions (for JIT)
@torch.jit.script
def q_multiply(q1, q2):
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ), dim=1)

@torch.jit.script
def q_to_rot_matrix(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2*(y**2 + z**2); r01 = 2*(x*y - z*w);      r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w);        r11 = 1 - 2*(x**2 + z**2); r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w);        r21 = 2*(y*z + x*w);       r22 = 1 - 2*(x**2 + y**2)
    
    row0 = torch.stack((r00, r01, r02), dim=1)
    row1 = torch.stack((r10, r11, r12), dim=1)
    row2 = torch.stack((r20, r21, r22), dim=1)
    return torch.stack((row0, row1, row2), dim=1)

@torch.jit.script
def rotate_vec(q, v):
    R = q_to_rot_matrix(q)
    return torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)

@torch.jit.script
def rotate_vec_inverse(q, v):
    R = q_to_rot_matrix(q)
    return torch.bmm(R.transpose(1, 2), v.unsqueeze(-1)).squeeze(-1)

# NumPy equivalents for EKF
def q_to_rot_matrix_numpy(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x**2+y**2)]
    ])

def rotate_vec_numpy(q, v): return q_to_rot_matrix_numpy(q) @ v
def rotate_vec_inverse_numpy(q, v): return q_to_rot_matrix_numpy(q).T @ v

# 2. MODEL CLASSES
# KALMAN NET
class KalmanNet(nn.Module):
    def __init__(self):
        super(KalmanNet, self).__init__()
        self.state_dim_pv = 6
        self.obs_dim = 6
        self.hidden_dim = 64
        self.nn_input_dim = self.obs_dim + 7 
        
        self.norm = nn.LayerNorm(self.nn_input_dim)
        self.gru = nn.GRUCell(self.nn_input_dim, self.hidden_dim)
        self.fc_k_gain = nn.Linear(self.hidden_dim, self.state_dim_pv * self.obs_dim)
        
        self.register_buffer('g_vec', torch.tensor([0.0, 0.0, 9.81]))
        self.register_buffer('mag_ref', torch.tensor([20.0, 0.0, -40.0]))
        self.register_buffer('dt', torch.tensor(DT))
        self.register_buffer('val_0_5', torch.tensor(0.5))

    def forward(self, x_curr, h_gru, inputs):
        accel = inputs[:, 0:3]; gyro = inputs[:, 3:6]; mag = inputs[:, 6:9]
        p, v, q = x_curr[:, 0:3], x_curr[:, 3:6], x_curr[:, 6:10]

        # 1. Prediction (Physics)
        half_ang = gyro * self.dt * self.val_0_5
        ones = torch.ones((x_curr.shape[0], 1), device=x_curr.device, dtype=x_curr.dtype)
        delta_q = torch.cat((ones, half_ang), dim=1)
        delta_q = delta_q / delta_q.norm(dim=1, keepdim=True)
        q_pred = q_multiply(q, delta_q)
        q_pred = q_pred / q_pred.norm(dim=1, keepdim=True)

        g_batch = self.g_vec.unsqueeze(0)
        mag_batch = self.mag_ref.unsqueeze(0)
        
        accel_world = rotate_vec(q, accel) - g_batch
        v_pred = v + accel_world * self.dt
        p_pred = p + v * self.dt + self.val_0_5 * accel_world * (self.dt ** 2)
        x_pred_full = torch.cat((p_pred, v_pred, q_pred), dim=1)

        # 2. Innovation
        g_body = rotate_vec_inverse(q_pred, g_batch)
        mag_body = rotate_vec_inverse(q_pred, mag_batch)
        y_pred = torch.cat((mag_body, g_body), dim=1)
        y_meas = torch.cat((mag, accel), dim=1)
        innovation = y_meas - y_pred

        # 3. Kalman Gain (NN)
        dynamic_state = x_pred_full[:, 3:10] 
        gru_in = torch.cat((innovation, dynamic_state), dim=1)
        gru_in = self.norm(gru_in)
        
        h_new = self.gru(gru_in, h_gru)
        K = self.fc_k_gain(h_new).reshape(-1, 6, 6)

        # 4. Correction (PV only)
        correction = torch.bmm(K, innovation.unsqueeze(2)).squeeze(2)
        p_new = p_pred + correction[:, 0:3]
        v_new = v_pred + correction[:, 3:6]
        
        x_new = torch.cat((p_new, v_new, q_pred), dim=1)
        return x_new, h_new

#B. BLACK BOX
class BlackBoxNavNet(nn.Module):
    def __init__(self):
        super(BlackBoxNavNet, self).__init__()
        self.input_dim = 9      # Acc(3) + Gyro(3) + Mag(3)
        self.output_dim = 10    # Pos(3) + Vel(3) + Quat(4)
        self.hidden_dim = 256   
        self.num_layers = 2     
        
        self.norm = nn.LayerNorm(self.input_dim)
        self.init_encoder = nn.Linear(self.output_dim, self.hidden_dim * self.num_layers)
        
        self.gru = nn.GRU(
            input_size=self.input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.0 # Dropout usually off for inference
        )
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def get_initial_hidden(self, init_state):
        # Maps (Batch, 10) -> (Num_Layers, Batch, Hidden)
        h0 = self.init_encoder(init_state)
        h0 = h0.view(self.num_layers, init_state.size(0), self.hidden_dim)
        return h0

    def forward_step(self, x_step, h_prev):
        x_norm = self.norm(x_step)
        gru_out, h_new = self.gru(x_norm, h_prev)
        
        # gru_out: (Batch, 1, Hidden)
        outputs = self.fc_out(gru_out)
        
        pos_vel = outputs[:, :, :6]
        quat_raw = outputs[:, :, 6:]
        quat_norm = F.normalize(quat_raw, p=2, dim=2)
        
        final_state = torch.cat((pos_vel, quat_norm), dim=2)
        return final_state, h_new

# 3. CLASSIC EKF 
class ClassicEKF:
    def __init__(self, R, Q):
        self.R = R
        self.Q = Q
        self.P = np.eye(10) * 0.1
        self.g_vec = G_VECTOR
        self.mag_ref = MAG_REF_WORLD

    def run_sequence(self, inputs, x_init):
        path = []
        x = x_init.copy()
        seq_len = len(inputs)
        
        for t in range(seq_len):
            gyro = inputs[t, 3:6]; accel = inputs[t, 0:3]; mag = inputs[t, 6:9]
            
            # PREDICT
            p, v, q = x[0:3], x[3:6], x[6:10]
            
            dq = np.concatenate(([1.0], gyro * DT * 0.5))
            dq /= np.linalg.norm(dq)
            
            w1, x1, y1, z1 = q; w2, x2, y2, z2 = dq
            q_pred = np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2
            ])
            q_pred /= np.linalg.norm(q_pred)
            
            R_mat = q_to_rot_matrix_numpy(q) 
            acc_world = R_mat @ accel - self.g_vec
            v_pred = v + acc_world * DT
            p_pred = p + v * DT + 0.5 * acc_world * DT**2
            
            x_pred = np.concatenate([p_pred, v_pred, q_pred])
            self.P[0:6, 0:6] += self.Q[0:6, 0:6]
            
            # UPDATE
            y_pred = np.concatenate([
                rotate_vec_inverse_numpy(q_pred, self.mag_ref),
                rotate_vec_inverse_numpy(q_pred, self.g_vec)
            ])
            
            innov = np.concatenate([mag, accel]) - y_pred
            H = np.zeros((6, 10))
            eps = 1e-4
            base_h = y_pred
            
            for i in range(6, 10):
                x_tmp = x_pred.copy()
                x_tmp[i] += eps
                x_tmp[6:10] /= np.linalg.norm(x_tmp[6:10]) 
                q_tmp = x_tmp[6:10]
                
                h_tmp = np.concatenate([
                    rotate_vec_inverse_numpy(q_tmp, self.mag_ref), 
                    rotate_vec_inverse_numpy(q_tmp, self.g_vec)
                ])
                H[:, i] = (h_tmp - base_h) / eps

            S = H @ self.P @ H.T + self.R
            try:
                K = self.P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                K = np.zeros((10, 6))

            x_new = x_pred + K @ innov
            x_new[6:10] /= np.linalg.norm(x_new[6:10])
            self.P = (np.eye(10) - K @ H) @ self.P
            x = x_new
            path.append(x)
            
        return np.array(path)

# 4. Main
def run_full_comparison():
    # 1. Load data
    try:
        inputs_raw = pd.read_csv('data_no_bias/test_inputs.csv', header=None).values
        targets_raw = pd.read_csv('data_no_bias/test_targets.csv', header=None).values
        print("Dataset loaded successfully.")
    except:
        print("Datasets not found. Generating dummy sequence...")
        inputs_raw = np.random.randn(1000, 9).astype(np.float32)
        targets_raw = np.zeros((1001, 10)); targets_raw[:, 6] = 1.0
        
    N_STEPS = min(len(inputs_raw), 1000) 
    inputs_raw = inputs_raw[:N_STEPS]
    targets_raw = targets_raw[:N_STEPS+1]
    
    # 2. Model Setup
    print("Setting up models...")
    
    # Classic EKF
    ekf_Q = np.eye(10) * 1e-3; ekf_Q[6:10] *= 1e-5
    ekf_R = np.eye(6); ekf_R[0:3] *= 1.0; ekf_R[3:6] *= 0.5
    ekf_model = ClassicEKF(ekf_R, ekf_Q)
    
    # KalmanNet
    knet_eager = KalmanNet().to(DEVICE)
    try:
        knet_eager.load_state_dict(torch.load("kalmannet_model_pv.pth", map_location=DEVICE))
        print("KNet Weights Loaded.")
    except:
        print("KNet Weights NOT found. Using Random.")
    knet_eager.eval()
    
    # KalmanNet (JIT)
    print("Compiling KNet (JIT)...")
    dummy_x = torch.zeros(1, 10, device=DEVICE); dummy_x[:, 6] = 1.0
    dummy_h = torch.zeros(1, 64, device=DEVICE)
    dummy_in = torch.zeros(1, 9, device=DEVICE)
    knet_jit = torch.jit.trace(knet_eager, (dummy_x, dummy_h, dummy_in))
    knet_jit = torch.jit.freeze(knet_jit) 
    
    # Black Box Model 
    print("Setting up Black Box...")
    bbox_model = BlackBoxNavNet().to(DEVICE)
    try:
        bbox_model.load_state_dict(torch.load("blackbox_best.pth", map_location=DEVICE))
        print("BlackBox Weights Loaded.")
    except:
        print("BlackBox Weights NOT found. Using Random.")
    bbox_model.eval()

    #3. Execution & Timing
    print("\nStarting Benchmark (Simulating Real-Time Step-by-Step)...")
    
    inputs_torch = torch.tensor(inputs_raw, device=DEVICE, dtype=torch.float32)
    init_state = targets_raw[0]
    
    results = { "Base": [], "EKF": [], "KNet_Eager": [], "KNet_JIT": [], "BlackBox": [] }
    latencies = { "Base": 0, "EKF": 0, "KNet_Eager": 0, "KNet_JIT": 0, "BlackBox": 0 }

    # Run Base (Dead Reckoning)
    start_t = time.perf_counter()
    base_ekf_helper = ClassicEKF(ekf_R, ekf_Q)
    base_ekf_helper.P = np.zeros((10,10)); base_ekf_helper.R = np.eye(6) * 1e9
    results["Base"] = base_ekf_helper.run_sequence(inputs_raw, init_state)
    latencies["Base"] = (time.perf_counter() - start_t) / N_STEPS * 1e6 

    # Run EKF
    start_t = time.perf_counter()
    results["EKF"] = ekf_model.run_sequence(inputs_raw, init_state)
    latencies["EKF"] = (time.perf_counter() - start_t) / N_STEPS * 1e6

    # Run KNet Eager
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start_t = time.perf_counter()
    path_eager = []
    with torch.no_grad():
        x_curr = torch.tensor(init_state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        h_curr = torch.zeros(1, 64, device=DEVICE)
        for t in range(N_STEPS):
            inp = inputs_torch[t].unsqueeze(0)
            x_curr, h_curr = knet_eager(x_curr, h_curr, inp)
            path_eager.append(x_curr.squeeze(0).cpu().numpy())
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    latencies["KNet_Eager"] = (time.perf_counter() - start_t) / N_STEPS * 1e6
    results["KNet_Eager"] = np.array(path_eager)

    # Run KNet JIT
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start_t = time.perf_counter()
    path_jit = []
    with torch.no_grad():
        x_curr = torch.tensor(init_state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        h_curr = torch.zeros(1, 64, device=DEVICE)
        for t in range(N_STEPS):
            inp = inputs_torch[t].unsqueeze(0)
            x_curr, h_curr = knet_jit(x_curr, h_curr, inp)
            path_jit.append(x_curr.squeeze(0).cpu().numpy())
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    latencies["KNet_JIT"] = (time.perf_counter() - start_t) / N_STEPS * 1e6
    results["KNet_JIT"] = np.array(path_jit)

    # Run Black Box 
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    start_t = time.perf_counter()
    path_bbox = []
    with torch.no_grad():
        # Initialize Hidden State ONE TIME
        init_tensor = torch.tensor(init_state, device=DEVICE, dtype=torch.float32).unsqueeze(0)
        h_curr = bbox_model.get_initial_hidden(init_tensor)
        
        for t in range(N_STEPS):
            # Input needs to be (Batch, Seq=1, Dim)
            inp = inputs_torch[t].unsqueeze(0).unsqueeze(0) 
            x_out, h_curr = bbox_model.forward_step(inp, h_curr)
            path_bbox.append(x_out.squeeze(0).squeeze(0).cpu().numpy())
            
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    latencies["BlackBox"] = (time.perf_counter() - start_t) / N_STEPS * 1e6
    results["BlackBox"] = np.array(path_bbox)

    # 4. Metrics & Console Output
    truth = targets_raw[1:N_STEPS+1]
    
    print("\n" + "="*75)
    print(f"{'Metric':<12} | {'Base':<8} | {'EKF':<8} | {'Eager':<8} | {'JIT':<8} | {'BlackBox':<8}")
    print("-" * 75)
    
    print(f"{'Lat (µs)':<12} | {latencies['Base']:<8.1f} | {latencies['EKF']:<8.1f} | {latencies['KNet_Eager']:<8.1f} | {latencies['KNet_JIT']:<8.1f} | {latencies['BlackBox']:<8.1f}")
    
    model_names = ["Base", "EKF", "KNet_Eager", "KNet_JIT", "BlackBox"]
    for name in model_names:
        pred = results[name]
        rmse = np.sqrt(np.mean((pred[:, 0:3] - truth[:, 0:3])**2))
        results[name + "_RMSE"] = rmse
        
    print(f"{'RMSE (m)':<12} | {results['Base_RMSE']:<8.3f} | {results['EKF_RMSE']:<8.3f} | {results['KNet_Eager_RMSE']:<8.3f} | {results['KNet_JIT_RMSE']:<8.3f} | {results['BlackBox_RMSE']:<8.3f}")
    print("-" * 75)

    # 5. Plot
    print(f"\nGenerating plots in {OUTPUT_DIR}...")
    
    # 1. Latency
    plt.figure(figsize=(10, 6))
    names = ['Base', 'EKF', 'KNet(E)', 'KNet(J)', 'BlackBox']
    times = [latencies['Base'], latencies['EKF'], latencies['KNet_Eager'], latencies['KNet_JIT'], latencies['BlackBox']]
    colors = ['gray', 'green', 'orange', 'blue', 'purple']
    bars = plt.bar(names, times, color=colors)
    plt.title(f"Inference Latency ({DEVICE})")
    plt.ylabel("µs per Step")
    for bar, t in zip(bars, times):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height(), f"{t:.0f}", ha='center', va='bottom')
    plt.savefig(f"{OUTPUT_DIR}/1_latency.png")
    plt.close()

    # 2. 3D Trajectory
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(truth[:,0], truth[:,1], truth[:,2], 'k-', linewidth=2, label='Ground Truth')
    ax.plot(results['Base'][:,0], results['Base'][:,1], results['Base'][:,2], 'r:', label='Base')
    ax.plot(results['EKF'][:,0], results['EKF'][:,1], results['EKF'][:,2], 'g--', label='EKF')
    ax.plot(results['KNet_JIT'][:,0], results['KNet_JIT'][:,1], results['KNet_JIT'][:,2], 'b-.', label='KNet')
    ax.plot(results['BlackBox'][:,0], results['BlackBox'][:,1], results['BlackBox'][:,2], color='purple', linestyle='-', alpha=0.7, label='BlackBox')
    ax.set_title("3D Trajectory Comparison")
    ax.legend()
    plt.savefig(f"{OUTPUT_DIR}/2_trajectory_3d.png")
    plt.close()

    # 3. Position Error
    plt.figure(figsize=(12, 5))
    err_ekf = np.linalg.norm(results['EKF'][:,:3]-truth[:,:3], axis=1)
    err_jit = np.linalg.norm(results['KNet_JIT'][:,:3]-truth[:,:3], axis=1)
    err_box = np.linalg.norm(results['BlackBox'][:,:3]-truth[:,:3], axis=1)
    
    plt.plot(err_ekf, 'g--', label='EKF')
    plt.plot(err_jit, 'b-', label='KNet')
    plt.plot(err_box, 'purple', alpha=0.8, label='BlackBox')
    plt.yscale('log')
    plt.title("Position Error over Time (Log Scale)")
    plt.ylabel("Error (m)")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.savefig(f"{OUTPUT_DIR}/3_error_log.png")
    plt.close()
    
    print("Done.")

if __name__ == "__main__":
    run_full_comparison()