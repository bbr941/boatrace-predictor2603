
import lightgbm as lgb
import pandas as pd
import json

def debug():
    bst = lgb.Booster(model_file='model_honmei.txt')
    feats = bst.feature_name()
    dump = bst.dump_model()
    pandas_cats = dump.get('pandas_categorical', []) # List of feature lists (values)? Or names?
    
    # In recent LightGBM, pandas_categorical might be list of [ [val1, val2], ... ] corresponding to cat features?
    # Or just list of names?
    # Actually, pandas_categorical is usually valid only if saved model has it.
    
    # Try to find categorical indices
    cat_indices = set()
    if 'categorical_feature' in dump:
        cat_indices = set(dump['categorical_feature'])
        
    print(f"Total Features: {len(feats)}")
    print("\n--- Categorical Features in Model ---")
    
    found_cats = []
    for i, f in enumerate(feats):
        is_cat = False
        if i in cat_indices:
            is_cat = True
            
        # Check against pandas_cats?
        # If pandas_cats is list of lists, we map by index in "categorical subset"?
        # It's complicated.
        # But if 'categorical_feature' is present, we trust it.
        
        if is_cat:
            found_cats.append(f)
            print(f" [CAT] {f}")
            
    print("\n--- Non-Categorical (Numeric) Features in Model ---")
    for f in feats:
        if f not in found_cats:
            print(f" [NUM] {f}")

if __name__ == "__main__":
    debug()
