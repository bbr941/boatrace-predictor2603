
import pandas as pd
import numpy as np
import lightgbm as lgb
import sqlite3
import train_model

def calc():
    # Load Data (5000 races)
    print("Loading...")
    df = pd.read_csv('boatrace_dataset_labeled_v2.csv')
    df = train_model.preprocess_data(df)
    
    # Filter DB Races
    conn = sqlite3.connect(r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT 5000")
    valid_races = {row[0] for row in cursor.fetchall()}
    conn.close()
    df = df[df['race_id'].isin(valid_races)].copy()
    
    # Predict
    model = lgb.Booster(model_file='model_honmei.txt')
    feats = model.feature_name()
    for f in feats:
        if f not in df.columns: df[f] = 0
    df['score'] = model.predict(df[feats])
    
    # Calc Std per race
    stats = df.groupby('race_id')['score'].std().reset_index()
    stats.columns = ['race_id', 'std']
    
    # Analyze Distribution
    median = stats['std'].median()
    p20 = stats['std'].quantile(0.20)
    
    print(f"\nScore Std Stats (N={len(stats)}):")
    print(f"Median (TH_HIGH): {median:.4f}")
    print(f"20th % (TH_LOW):  {p20:.4f}")
    
if __name__ == "__main__":
    calc()
