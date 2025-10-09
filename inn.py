import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
import joblib
import ast
import os
import FrEIA.framework as Ff
import FrEIA.modules as Fm

# =================================================================
# === 1. CONFIGURATION
# =================================================================

CSV_FILE_PATH = 'Forward/final_pts.csv' 
TEST_SPLIT_FRACTION = 0.2
MODEL_SAVE_PATH = 'best_model.pth'
X_SCALER_PATH = 'X_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'

# --- Model & Training Parameters ---
N_DIMS_INPUT = 6
N_DIMS_OUTPUT = 3
N_DIMS_TOTAL = N_DIMS_INPUT


N_EPOCHS = 50
BATCH_SIZE = 32 # Increased batch size slightly for faster training on large data
LEARNING_RATE = 2e-4

# Loss Weights
LAMBDA_RECON = 5.0
LAMBDA_LATENT = 0.1
LAMBDA_JACOBIAN = 1.0

# =================================================================
# === 2. DATA HANDLING
# =================================================================

class TrajectoryDataset(Dataset):
    """Custom PyTorch Dataset to load and parse trajectory data from a CSV."""
    def __init__(self, csv_file):
        try:
            df = pd.read_csv(csv_file)
            df.columns = df.columns.str.strip()
        except Exception as e:
            print(f"FATAL ERROR: Could not read the file '{csv_file}'. Error: {e}")
            exit()
            
        input_cols = ['p_x', 'p_y', 'p_z', 'v_mag', 'phi', 'w_y']
        output_col = 'p_f'
        
        try:
            self.x = df[input_cols].values.astype(np.float32)
            self.y = np.array([ast.literal_eval(i) for i in df[output_col]]).astype(np.float32)
        except KeyError as e:
            print(f"FATAL KeyError: A required column was not found: {e}")
            print(f"Columns the script found in your file: {list(df.columns)}")
            print("Please ensure your CSV header matches the 'input_cols' list exactly.")
            exit()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# =================================================================
# === 3. MODEL DEFINITION
# =================================================================

def build_inn_model():
    def subnet_fc(c_in, c_out):
        return nn.Sequential(nn.Linear(c_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, c_out))
    inn = Ff.SequenceINN(N_DIMS_TOTAL)
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    return inn

# =================================================================
# === 4. MAIN SCRIPT EXECUTION
# =================================================================

if __name__ == "__main__":
    full_dataset = TrajectoryDataset(CSV_FILE_PATH)
    print(f"Loaded {len(full_dataset)} records from '{CSV_FILE_PATH}'.")

    test_size = int(len(full_dataset) * TEST_SPLIT_FRACTION)
    train_size = len(full_dataset) - test_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    print(f"Splitting data: {train_size} for training, {test_size} for testing.")

    X_scaler = StandardScaler().fit(train_dataset.dataset.x[train_dataset.indices])
    y_scaler = StandardScaler().fit(train_dataset.dataset.y[train_dataset.indices])
    
    joblib.dump(X_scaler, X_SCALER_PATH)
    joblib.dump(y_scaler, Y_SCALER_PATH)
    print(f"Data scalers saved to '{X_SCALER_PATH}' and '{Y_SCALER_PATH}'.")
    
    full_dataset.x = X_scaler.transform(full_dataset.x)
    full_dataset.y = y_scaler.transform(full_dataset.y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = build_inn_model()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("\nStarting model training...")
    best_val_loss = float('inf')
    for epoch in range(N_EPOCHS):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            z_pred, log_jac_det = model(x_batch)
            y_pred, z_latent = z_pred[:, :N_DIMS_OUTPUT], z_pred[:, N_DIMS_OUTPUT:]
            loss_recon = torch.mean((y_pred - y_batch)**2)
            loss_latent = 0.5 * torch.mean(z_latent**2)
            loss_jacobian = torch.mean(-log_jac_det)
            total_loss = (LAMBDA_RECON * loss_recon + LAMBDA_LATENT * loss_latent + LAMBDA_JACOBIAN * loss_jacobian)
            total_loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in test_loader:
                z_val, _ = model(x_val)
                y_pred_val = z_val[:, :N_DIMS_OUTPUT]
                val_loss += torch.mean((y_pred_val - y_val)**2).item()
        
        val_loss /= len(test_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        
        print(f"Epoch {epoch+1}/{N_EPOCHS} | Val Loss: {val_loss:.6f} {'(New best model saved)' if val_loss == best_val_loss else ''}")


    print(f"\nTraining finished. Best model saved to '{MODEL_SAVE_PATH}'.")

