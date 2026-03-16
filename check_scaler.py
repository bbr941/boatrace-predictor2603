import pickle
import os

MODEL_DIR = 'models'
scaler_path = os.path.join(MODEL_DIR, 'alt_scaler.pkl')

if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Features: {list(scaler.feature_names_in_)}")
    else:
        print("No feature_names_in_ attribute.")
else:
    print("scaler not found")
