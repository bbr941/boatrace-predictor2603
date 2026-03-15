
import pandas as pd
import lightgbm as lgb
import numpy as np
import os
import train_model

MODEL_ANA = 'model_ana.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'

def evaluate_ana_metrics():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess (Categorical etc) - Reuse train_model logic
    df = train_model.preprocess_data(df)
    
    # Split (Same logic as train)
    unique_races = df['race_id'].unique()
    split_idx = int(len(unique_races) * 0.8)
    test_races = unique_races[split_idx:]
    test_df = df[df['race_id'].isin(test_races)].copy()
    
    print(f"Test Set: {len(test_races)} races, {len(test_df)} rows")
    
    # Load Model
    if not os.path.exists(MODEL_ANA):
        print(f"Model {MODEL_ANA} not found.")
        return
        
    model = lgb.Booster(model_file=MODEL_ANA)
    feats = model.feature_name()
    
    # Ensure features exist
    for f in feats:
        if f not in test_df.columns:
            test_df[f] = 0
            
    # Predict
    print("Predicting...")
    preds = model.predict(test_df[feats])
    test_df['score'] = preds
    
    # Evaluation
    # 1. Hit Rate (Top 1)
    # 2. Recovery Rate (Top 1)
    
    # Group by race -> Find max score boat -> Check rank
    results = []
    
    # Check for odds column
    has_odds = 'odds' in test_df.columns
    has_syn = 'syn_win_rate' in test_df.columns
    
    if not has_odds and not has_syn:
        print("No Odds information available for Recovery Rate.")
        return
        
    print(f"Using Odds Source: {'odds' if has_odds else '1/syn_win_rate'}")

    hits = 0
    total_races = 0
    total_return = 0
    total_bet = 0
    
    groups = test_df.groupby('race_id')
    
    for rid, group in groups:
        # Get boat with max score
        top_boat = group.loc[group['score'].idxmax()]
        
        # Bet 100 yen
        total_bet += 100
        total_races += 1
        
        # Check Win
        if top_boat['rank'] == 1:
            hits += 1
            # Payout
            if has_odds:
                odds = top_boat['odds']
            else:
                 # Proxy from syn_win_rate
                 swr = top_boat['syn_win_rate']
                 odds = 1.0 / swr if swr > 0 else 1.0
                 # Apply a typical "Take" margin? 
                 # Boat race return rate is ~75%. 
                 # syn_win_rate is prob. Odds ~= 0.75 / prob?
                 # Or assume calculated odds_1min from DB was already odds.
                 # syn_win_rate = 1/odds_1min. So 1/syn = odds_1min.
                 
            payout = odds * 100
            total_return += payout
            
    hit_rate = hits / total_races
    recovery_rate = total_return / total_bet
    
    print("\n=== Ana Model Performance (Test Set) ===")
    print(f"Total Races: {total_races}")
    print(f"Hit Rate (Tansho): {hit_rate:.2%}")
    print(f"Recovery Rate (Tansho): {recovery_rate:.2%}")
    print(f"Total Bet: {total_bet}, Total Return: {int(total_return)}")

if __name__ == "__main__":
    evaluate_ana_metrics()
