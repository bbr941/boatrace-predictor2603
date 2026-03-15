
import lightgbm as lgb
import pandas as pd
import numpy as np
import json

def debug():
    print("Loading Model...")
    bst = lgb.Booster(model_file='model_honmei.txt')
    
    # Get ID of categorical features
    # LightGBM saves categorical feature indices.
    # We can get feature names.
    feat_names = bst.feature_name()
    
    # To find which are categorical, we can inspect the model dump
    dump = bst.dump_model()
    
    # 'pandas_categorical' might be present if trained with pandas dataframe
    pandas_cats = dump.get('pandas_categorical', [])
    
    print(f"Model Feature Count: {len(feat_names)}")
    if pandas_cats:
        print("Model Expects Categorical Features (Pandas):")
        for c in pandas_cats:
            # pandas_categorical is usually list of lists or valid format??
            # It's usually a list of column names if stored?
            # Or list of [index, name]?
            print(f" - {c}")
    else:
        print("No 'pandas_categorical' info found in dump. Checking 'feature_info'...")
        # 'feature_info' might have min/max/type?
        pass

    print("-" * 30)
    
    # Now Check feature_names closely
    # Let's see if we can infer likely categories or check against train_model logic
    for f in feat_names:
        if f in ['class', 'racer_class', 'branch', 'wind_direction', 'weather', 'venue_name']:
            print(f"Feature '{f}' is in Model.")
            
    # Also check if 'racer_id' is used?
    if 'racer_id' in feat_names: print("'racer_id' is in Model.")
    
    # Categorical indices
    # Only way to know for sure in LGBM Booster (without Metadata) is hard.
    # But the error "train and valid dataset categorical_feature do not match" usually comes from:
    # 1. Using Pandas DataFrame for prediction
    # 2. Columns having 'category' dtype in DF
    # 3. Model remembering which columns were 'category' during training.
    
    # We will simulate the App's DF creation and compare dtypes.
    
    # Dummy App Data
    # Mimic app_boatrace.py processing
    df = pd.DataFrame({
        'race_id': ['1'],
        'race_date': ['20240101'],
        'venue_name': ['桐生'],
        'boat_number': [1],
        'class': ['A1'],
        'branch': ['東京'],
        'wind_direction': ['北'],
        'weather': ['曇り'],
        'racer_id': [4000],
        'exhibition_time': [6.7],
         # Add numeric cols that app converts
        'nige_count': [10],
        'course_run_count': [20]
    })
    
    # App Logic (Simplified)
    # 1. Numeric Conversion (Exclusions)
    # exclusions = ['race_id', 'race_date', 'venue_name', 'prior_results', 'wind_direction', 'branch', 'class', 'racer_class']
    # If weather is NOT excluded, it becomes NaN (Numeric).
    
    # Check 'weather'!
    # Is 'weather' in Model?
    # If 'weather' is in Model, and train_model kept it as Category...
    # Then App converts it to Numeric (NaN) -> Mismatch.
    
    # Check 'place' (venue_name)?
    # venue_name is mapped to code.
    
    print("\n--- Checking Potential Culprits ---")
    if 'weather' in feat_names:
        print("[CRITICAL] 'weather' is a Model Feature.")
    else:
        print("'weather' is NOT in Model.")

if __name__ == "__main__":
    debug()
