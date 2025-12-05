import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm

# config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# model
class BlackBoxNavNet(nn.Module):
    def __init__(self):
        super(BlackBoxNavNet, self).__init__()
        
        # Dimensions
        self.input_dim = 9      # Acc(3) + Gyro(3) + Mag(3)
        self.output_dim = 10    # Pos(3) + Vel(3) + Quat(4)
        self.hidden_dim = 256   # Needs more capacity to learn physics
        self.num_layers = 2     # Stacking layers for complex dynamics
        
        # Input Normalization
        self.norm = nn.LayerNorm(self.input_dim)
        
        # 1. Initial State Encoder
        self.init_encoder = nn.Linear(self.output_dim, self.hidden_dim * self.num_layers)
        
        # 2. Main Recurrent Backbone
        self.gru = nn.GRU(
            input_size=self.input_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers,
            batch_first=True,
            dropout=0.1
        )
        
        # 3. Output Head
        self.fc_out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x, init_state):
        # x shape: (Batch, Seq_Len, 9)
        # init_state shape: (Batch, 10)
        
        # Normalize inputs
        x = self.norm(x)
        
        # initialize GRU hidden state
        # Shape for GRU: (num_layers, batch, hidden_dim)
        h0 = self.init_encoder(init_state)
        h0 = h0.view(self.num_layers, x.size(0), self.hidden_dim)
        
        # Run GRU
        gru_out, _ = self.gru(x, h0)
        
        # Map to State Space
        outputs = self.fc_out(gru_out)
        
        # Split outputs 
        pos_vel = outputs[:, :, :6]
        quat_raw = outputs[:, :, 6:]
        
        # Enforce unit quaternions
        quat_norm = F.normalize(quat_raw, p=2, dim=2)
        # get final state
        final_state = torch.cat((pos_vel, quat_norm), dim=2)
        
        return final_state

# data loading
def load_split_data(split_name, seq_len=100):
    # Ensure this path matches your folder structure
    input_path = f"data_no_bias/{split_name}_inputs.csv"
    target_path = f"data_no_bias/{split_name}_targets.csv"
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return None, None, None

    inputs_raw = pd.read_csv(input_path, header=None).values
    targets_raw = pd.read_csv(target_path, header=None).values
    # Shift the data for desire
    inputs_shifted = inputs_raw[:-1]
    targets_shifted = targets_raw[1:]
    init_states = targets_raw[:-1] 

    n_samples = (len(inputs_shifted) // seq_len) * seq_len
    
    X = inputs_shifted[:n_samples].reshape(-1, seq_len, 9)
    Y = targets_shifted[:n_samples].reshape(-1, seq_len, 10)
    Init = init_states[:n_samples].reshape(-1, seq_len, 10)
    
    # only need the very first state to initialize the RNN
    Init_Start = Init[:, 0, :] 

    return (torch.tensor(X, dtype=torch.float32).to(device),
            torch.tensor(Y, dtype=torch.float32).to(device),
            torch.tensor(Init_Start, dtype=torch.float32).to(device))

# loop
def train_blackbox():
    SEQ_LEN = 100 # Only need short since we want the filter for short term filtering
    train_data = load_split_data('train', SEQ_LEN)
    val_data = load_split_data('val', SEQ_LEN)
    
    if train_data[0] is None: 
        print("Data not found. Please ensure 'data_no_bias' folder exists.")
        return

    X_train, Y_train, Init_train = train_data
    X_val, Y_val, Init_val = val_data

    model = BlackBoxNavNet().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # Loss: MSE. 
    criterion = nn.MSELoss()

    print(f"Starting Black Box Training...")
    print(f"Train Shape: {X_train.shape}")
    
    epochs = 100
    progress_bar = tqdm(range(epochs), desc="Training")
    
    best_val_loss = float('inf')

    for epoch in progress_bar:
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train, Init_train)
        
        loss = criterion(outputs, Y_train)
        loss.backward()
        
        # Gradient clipping for prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_out = model(X_val, Init_val)
            val_loss = criterion(val_out, Y_val)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "blackbox_best.pth")
        
        progress_bar.set_postfix({'Train': f'{loss.item():.4f}', 'Val': f'{val_loss.item():.4f}'})

    print("\nTraining Complete.")
    
    #Evalution
    print("Running Evaluation on Test Set...")
    try:
        test_inputs = pd.read_csv('data_no_bias/test_inputs.csv', header=None).values
        test_targets = pd.read_csv('data_no_bias/test_targets.csv', header=None).values
        
        limit = 1000
        t_in = torch.tensor(test_inputs[:limit], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Initial state from the ground truth start
        t_init = torch.tensor(test_targets[0], dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            pred = model(t_in, t_init)
            pred_np = pred.squeeze(0).cpu().numpy()
            
        truth = test_targets[1:limit+1] # Align targets
        
        # RMSE Calculation
        min_len = min(len(pred_np), len(truth))
        pred_np = pred_np[:min_len]
        truth = truth[:min_len]

        rmse_pos = np.sqrt(np.mean((pred_np[:, 0:3] - truth[:, 0:3])**2, axis=0))
        rmse_vel = np.sqrt(np.mean((pred_np[:, 3:6] - truth[:, 3:6])**2, axis=0))
        
        print(f"Test RMSE (Pos X): {rmse_pos[0]:.4f} m")
        print(f"Test RMSE (Pos Y): {rmse_pos[1]:.4f} m")
        print(f"Test RMSE (Vel X): {rmse_vel[0]:.4f} m/s")
        
    except Exception as e:
        print(f"Could not run test evaluation: {e}")

if __name__ == "__main__":
    train_blackbox()