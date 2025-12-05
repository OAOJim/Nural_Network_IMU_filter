import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm

# Config
DT = 0.01
# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# math functions
def q_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=1)

def q_to_rot_matrix(q):
    """Converts a batch of quaternions to rotation matrices."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2*(y**2 + z**2); r01 = 2*(x*y - z*w);      r02 = 2*(x*z + y*w)
    r10 = 2*(x*y + z*w);        r11 = 1 - 2*(x**2 + z**2); r12 = 2*(y*z - x*w)
    r20 = 2*(x*z - y*w);        r21 = 2*(y*z + x*w);       r22 = 1 - 2*(x**2 + y**2)
    
    row0 = torch.stack((r00, r01, r02), dim=1)
    row1 = torch.stack((r10, r11, r12), dim=1)
    row2 = torch.stack((r20, r21, r22), dim=1)
    return torch.stack((row0, row1, row2), dim=1)

def rotate_vec(q, v):
    """Rotates vector v by quaternion q."""
    R = q_to_rot_matrix(q) 
    v_expanded = v.unsqueeze(-1) 
    v_rotated = torch.bmm(R, v_expanded)
    return v_rotated.squeeze(-1)

def rotate_vec_inverse(q, v):
    """Rotates vector v by the inverse of quaternion q (World -> Body)."""
    R = q_to_rot_matrix(q)
    v_expanded = v.unsqueeze(-1)
    v_rotated = torch.bmm(R.transpose(1, 2), v_expanded)
    return v_rotated.squeeze(-1)

# Model
class KalmanNet(nn.Module):
    def __init__(self):
        super(KalmanNet, self).__init__()
        
        # State & Observation
        self.state_dim_full = 10 # Pos(3), Vel(3), Quat(4)
        self.state_dim_pv = 6    # Pos(3), Vel(3) - States corrected by NN
        self.obs_dim = 6         # Mag(3) + Accel(3) - Inputs to NN
        self.hidden_dim = 64
        
        # NN Input: Innovation (6) + Velocity (3) + Quaternion (4) = 13 inputs
        self.nn_input_dim = self.obs_dim + 7 
        
        # Normalization Layer
        self.norm = nn.LayerNorm(self.nn_input_dim)

        # GRU
        self.gru = nn.GRUCell(input_size=self.nn_input_dim, hidden_size=self.hidden_dim) 
        
        # Output Layer: Predicts K for PV state (6 x 6 matrix)
        self.fc_k_gain = nn.Linear(self.hidden_dim, self.state_dim_pv * self.obs_dim) 

        # Weight init
        nn.init.uniform_(self.fc_k_gain.weight, -0.001, 0.001)
        nn.init.zeros_(self.fc_k_gain.bias)
        
        # Physics Constants
        self.g_vec = torch.tensor([0.0, 0.0, 9.81], device=device)
        self.mag_ref = torch.tensor([20.0, 0.0, -40.0], device=device)

    def forward(self, inputs, initial_state_tuple=None, q_gt_seq=None):
        batch_size, seq_len, _ = inputs.shape
        
        # Constants for batch
        g_vec_batch = self.g_vec.unsqueeze(0).expand(batch_size, -1)
        mag_ref_batch = self.mag_ref.unsqueeze(0).expand(batch_size, -1)

        # Initialize State
        if initial_state_tuple is None:
            x_curr = torch.zeros(batch_size, self.state_dim_full, device=device)
            x_curr[:, 6] = 1.0 # unit quaternion
            h_gru = torch.zeros(batch_size, self.hidden_dim, device=device)
        else:
            x_curr, h_gru = initial_state_tuple

        outputs = []

        # Loop
        for t in range(seq_len):
            # 1. Sensor Inputs
            accel_meas = inputs[:, t, 0:3]
            gyro_meas = inputs[:, t, 3:6]
            mag_meas = inputs[:, t, 6:9]

            # 2. Prediction (math)
            p = x_curr[:, 0:3]
            v = x_curr[:, 3:6]
            q = x_curr[:, 6:10] # Estimated q_t

            if q_gt_seq is not None:
                q_for_rotation = q_gt_seq[:, t, :]
            else:
                q_for_rotation = q 
            
            # 2a. Orientation Integration 
            half_ang = gyro_meas * DT * 0.5
            ones = torch.ones(batch_size, 1, device=device)
            delta_q = torch.cat((ones, half_ang), dim=1)
            delta_q = delta_q / torch.norm(delta_q, dim=1, keepdim=True)
            
            q_pred = q_multiply(q, delta_q)
            q_pred = q_pred / torch.norm(q_pred, dim=1, keepdim=True)

            # 2b. Position/Velocity Integration
            # CRITICAL CHANGE: Use q_for_rotation (ground truth q_t)
            accel_world = rotate_vec(q_for_rotation, accel_meas) - g_vec_batch
            
            v_pred = v + accel_world * DT
            p_pred = p + v * DT + 0.5 * accel_world * DT**2
            
            x_pred_pv = torch.cat((p_pred, v_pred), dim=1)
            x_pred_full = torch.cat((x_pred_pv, q_pred), dim=1)

            # 3. observation prediction
            g_body_pred = rotate_vec_inverse(q_pred, g_vec_batch)
            mag_body_pred = rotate_vec_inverse(q_pred, mag_ref_batch)
            
            y_meas = torch.cat((mag_meas, accel_meas), dim=1)
            y_pred = torch.cat((mag_body_pred, g_body_pred), dim=1)

            # 4. Innovation
            innovation = y_meas - y_pred 
            
            # 5. NN Kalman gain calculation
            dynamic_state = x_pred_full[:, 3:10] # V(3) + Q(4)
            gru_in = torch.cat((innovation, dynamic_state), dim=1)
            
            gru_in = self.norm(gru_in)
            h_gru = self.gru(gru_in, h_gru)
            
            # Predict Gain K 
            k_flat = self.fc_k_gain(h_gru)
            K = k_flat.reshape(batch_size, self.state_dim_pv, self.obs_dim) # 6x6
            
            # 6. update
            innovation_expanded = innovation.unsqueeze(2) 
            
            # Calculate correction for PV states 
            correction_pv = torch.bmm(K, innovation_expanded).squeeze(2)
            
            # new full state
            p_new = x_pred_pv[:, 0:3] + correction_pv[:, 0:3]
            v_new = x_pred_pv[:, 3:6] + correction_pv[:, 3:6]
            
            q_final = q_pred 
            
            x_curr = torch.cat((p_new, v_new, q_final), dim=1)
            
            outputs.append(x_curr)

        return torch.stack(outputs, dim=1)

# Data loading
def load_split_data(split_name, seq_len=100):
    input_path = f"data_no_bias/{split_name}_inputs.csv"
    target_path = f"data_no_bias/{split_name}_targets.csv"
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return None, None, None

    inputs_raw = pd.read_csv(input_path, header=None).values
    targets_raw = pd.read_csv(target_path, header=None).values

    inputs_shifted = inputs_raw[:-1]
    targets_shifted = targets_raw[1:]
    init_states = targets_raw[:-1]

    n_samples = (len(inputs_shifted) // seq_len) * seq_len
    
    X = inputs_shifted[:n_samples].reshape(-1, seq_len, 9)
    Y = targets_shifted[:n_samples].reshape(-1, seq_len, 10)
    Init = init_states[:n_samples].reshape(-1, seq_len, 10)
    
    Init_Start = Init[:, 0, :]

    return (torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(Y, dtype=torch.float32).to(device),
            torch.tensor(Init_Start, dtype=torch.float32).to(device))

# Main
def train():
    SEQ_LEN = 50
    train_data = load_split_data('train', SEQ_LEN)
    val_data = load_split_data('val', SEQ_LEN)
    
    if train_data[0] is None: return

    X_train, Y_train, Init_train = train_data
    X_val, Y_val, Init_val = val_data

    model = KalmanNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()

    print(f"Starting Training: Train Batch {X_train.shape}, Val Batch {X_val.shape}")
    
    # Construct Ground Truth Quaternion Sequences for training (q_t sequence)
    q_gt_0 = Init_train[:, 6:10].unsqueeze(1)
    q_gt_1_to_L_minus_1 = Y_train[:, :-1, 6:10]
    q_gt_train_seq = torch.cat((q_gt_0, q_gt_1_to_L_minus_1), dim=1)
    
    # Construct Ground Truth Quaternion Sequences for validation
    q_gt_val_0 = Init_val[:, 6:10].unsqueeze(1)
    q_gt_val_1_to_L_minus_1 = Y_val[:, :-1, 6:10]
    q_gt_val_seq = torch.cat((q_gt_val_0, q_gt_val_1_to_L_minus_1), dim=1)
    
    epochs = 100
    progress_bar = tqdm(range(epochs), desc="Training")
    
    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        h_init = torch.zeros(len(X_train), 64, device=device)
        # Pass ground truth quaternion sequence for training
        outputs = model(X_train, initial_state_tuple=(Init_train, h_init), q_gt_seq=q_gt_train_seq)
        
        # Loss calculated on the FULL 10-state vector 
        loss = criterion(outputs, Y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            h_val_init = torch.zeros(len(X_val), 64, device=device)
            # Pass ground truth quaternion sequence for validation
            val_out = model(X_val, initial_state_tuple=(Init_val, h_val_init), q_gt_seq=q_gt_val_seq)
            val_loss = criterion(val_out, Y_val)
        
        progress_bar.set_postfix({'Train': f'{loss.item():.4f}', 'Val': f'{val_loss.item():.4f}'})

    # Save Model
    save_path = "kalmannet_model_pv.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nModel saved to {save_path}")
    
    # Quick Test
    print("\nRunning Quick Sanity Check (First 10s of Test)...")
    test_inputs = pd.read_csv('data_no_bias/test_inputs.csv', header=None).values[:1000] # First 1000 steps
    test_targets = pd.read_csv('data_no_bias/test_targets.csv', header=None).values[:1000]
    
    with torch.no_grad():
        full_input = torch.tensor(test_inputs[:-1], dtype=torch.float32).unsqueeze(0).to(device)
        start_true = torch.tensor(test_targets[0], dtype=torch.float32).unsqueeze(0).to(device)
        h_start = torch.zeros(1, 64, device=device)
        
        # In inference mode, q_gt_seq is None
        full_pred = model(full_input, initial_state_tuple=(start_true, h_start), q_gt_seq=None)
        full_pred_np = full_pred.squeeze(0).cpu().numpy()
        
        truth = test_targets[1:]
        # Calculate RMSE
        pv_rmse = np.sqrt(np.mean((full_pred_np[:, 0:6] - truth[:, 0:6])**2, axis=0))
        
        print(f"Test RMSE (Pos X): {pv_rmse[0]:.4f} m")
        print(f"Test RMSE (Vel X): {pv_rmse[3]:.4f} m/s")

if __name__ == "__main__":
    train()