
import pandas as pd
import numpy as np
import lightgbm as lgb
import sys
import os

# Import train_model logic
import train_model

MODEL_HONMEI = 'model_honmei.txt'

def get_model_features(path):
    print(f"--- Loading Model: {path} ---")
    if not os.path.exists(path):
        print("Model file not found.")
        return []
    model = lgb.Booster(model_file=path)
    feats = model.feature_name()
    print(f"Model Expects {len(feats)} features.")
    return feats


def get_app_generated_features(all_columns):
    print("\n--- Generating App Features ---")
    data = {c: [1] for c in all_columns}
    # Add extras found in App
    extras = ['wind_angle_deg', 'venue_tailwind_deg', 'venue_code_int', 'weather', 'boat_before'] # and others?
    for e in extras: data[e] = [1]
    
    df = pd.DataFrame(data)
    
    # --- COPY OF APP LOGIC (get_features_subset) ---
    base_ignore = [
        'race_id', 'boat_number', 'racer_id', 'rank', 'relevance',
        'race_date', 'venue_name', 'prior_results', 'weight_for_loss', 'pred_score', 'score',
        # Extra columns created in app but not in training
        'wind_angle_deg', 'venue_tailwind_deg', 'venue_code_int',
        'weather', 'boat_before', 'params'
    ]
    
    all_cols = df.columns.tolist()
    candidates = [c for c in all_cols if c not in base_ignore]
    
    # Also filter out 'weather', 'boat_before' if they are in df?
    # In reality, 'weather' etc are dicts, not columns in df (unless normalized).
    # Assuming df only has columns that are features (or intermediate columns).
    
    # LightGBM requires EXACT match of features.
    # If `candidates` has extra stuff, it fails.
    
    return candidates

if __name__ == "__main__":
    model_feats = get_model_features(MODEL_HONMEI)
    
    # Headers from CSV
    try:
        df_headers = pd.read_csv('boatrace_dataset_labeled_v2.csv', nrows=1).columns.tolist()
    except:
        df_headers = [] # fallback if file not found
        
    app_feats = get_app_generated_features(df_headers)
    
    print(f"Model Features: {len(model_feats)}")
    print(f"App Features: {len(app_feats)}")
    
    diff = set(app_feats) - set(model_feats)
    if diff:
        print(f"Mismatch: App has {len(diff)} extra features: {diff}")
    else:
        print("Success: App features match Model features.")
