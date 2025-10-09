import torch
import numpy as np
import joblib
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import torch.nn as nn

# =================================================================
# === 1. CONFIGURATION
# =================================================================
MODEL_PATH = 'best_model.pth'
X_SCALER_PATH = 'X_scaler.joblib'
Y_SCALER_PATH = 'y_scaler.joblib'

N_DIMS_INPUT = 6
N_DIMS_OUTPUT = 3
N_DIMS_TOTAL = N_DIMS_INPUT

# --- Define Your Custom Input ---
custom_pf_input =[17.158298961035513, -0.9473684210526315, -0.05259151767745186]

# =================================================================
# === 2. MODEL DEFINITION (This now correctly matches the training script)
# =================================================================
def build_inn_model():
    """Defines the exact same architecture as used in training."""
    def subnet_fc(c_in, c_out):
        # ==> THE FIX IS HERE: The second Linear layer now correctly takes 128 inputs <==
        return nn.Sequential(nn.Linear(c_in, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, c_out))
    
    inn = Ff.SequenceINN(N_DIMS_TOTAL)
    for k in range(8):
        inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc, permute_soft=True)
    return inn

# =================================================================
# === 3. PREDICTION SCRIPT
# =================================================================
if __name__ == "__main__":
    try:
        model = build_inn_model()
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        X_scaler = joblib.load(X_SCALER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        print("Successfully loaded model and data scalers.")
    except FileNotFoundError as e:
        print(f"FATAL ERROR: Could not find a required file: {e}")
        print("Please run the 'train_model.py' script first to create these files.")
        exit()
    except RuntimeError as e:
        print(f"FATAL RuntimeError: {e}")
        print("\nThis usually means the model architecture in this script does not match the one in the saved '.pth' file.")
        print("Please ensure the 'build_inn_model' function is identical to the one in your training script.")
        exit()

    
    with torch.no_grad():
        pf_np = np.array(custom_pf_input).reshape(1, -1)
        pf_scaled = y_scaler.transform(pf_np)
        pf_tensor = torch.tensor(pf_scaled, dtype=torch.float32)
        
        latent_vars = torch.randn(1, N_DIMS_TOTAL - N_DIMS_OUTPUT)
        model_input = torch.cat([pf_tensor, latent_vars], dim=1)
        
        generated_output_scaled, _ = model(model_input, rev=True)
        generated_features = X_scaler.inverse_transform(generated_output_scaled.numpy())

    print("\n" + "="*50)
    print("         REVERSE INFERENCE RESULTS")
    print("="*50)
    print(f"\nInput Final Position (p_f):\n{custom_pf_input}")
    
    print("\nCALCULATED Initial Conditions:")
    feature_names = ['p_x', 'p_y', 'p_z', 'v_mag', 'phi', 'w_y']
    for name, value in zip(feature_names, generated_features[0]):
        print(f"  - {name:<8}: {value:.4f}")
    print("\n" + "="*50)

