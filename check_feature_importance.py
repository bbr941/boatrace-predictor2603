
import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import os

MODELS = {'Honmei': 'model_honmei.txt', 'Ana': 'model_ana.txt'}

def check_importance():
    for name, path in MODELS.items():
        if not os.path.exists(path):
            continue
            
        print(f"\n--- {name} Model Feature Importance ---")
        model = lgb.Booster(model_file=path)
        
        importance = model.feature_importance(importance_type='gain')
        names = model.feature_name()
        
        df_imp = pd.DataFrame({'feature': names, 'gain': importance})
        df_imp = df_imp.sort_values('gain', ascending=False).head(20)
        
        print(df_imp)
        
        # Check for suspicious features
        suspicious = ['rank', 'finish_order', 'race_time', 'result', 'prize']
        found = [f for f in names if any(s in f for s in suspicious)]
        if found:
            print(f"⚠️ WARNING: Suspicious features found in model: {found}")
        else:
            print("No obviously suspicious columns (rank, time) found in features.")

if __name__ == "__main__":
    check_importance()
