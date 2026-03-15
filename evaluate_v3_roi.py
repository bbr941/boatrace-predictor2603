import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import pickle
import os
import itertools
from sklearn.metrics import log_loss

# --- Configuration ---
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
MODEL_DIR = 'models'

def load_data(limit=10000):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
    SELECT
        re.race_id, re.boat_number, re.racer_id, r.venue_code,
        bi.exhibition_time, bi.exhibition_start_timing, 
        COALESCE(bi.exhibition_entry_course, re.boat_number) as pred_course,
        re.nat_win_rate, re.motor_rate, re.boat_rate,
        res.finish_order as rank
    FROM race_entries re
    JOIN races r ON re.race_id = r.race_id
    LEFT JOIN before_info bi ON re.race_id = bi.race_id AND re.boat_number = bi.boat_number
    LEFT JOIN results res ON re.race_id = res.race_id AND re.boat_number = res.boat_number
    WHERE res.finish_order IS NOT NULL
    ORDER BY r.race_date DESC, re.race_id DESC
    LIMIT {limit * 6}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Preprocessing
    df['target'] = (df['rank'] == 1).astype(int)
    df['racer_id'] = df['racer_id'].fillna(0).astype(int).astype(str).astype('category')
    df['venue_code'] = df['venue_code'].fillna(0).astype(int).astype(str).astype('category')
    num_cols = ['exhibition_time', 'exhibition_start_timing', 'nat_win_rate', 'motor_rate', 'boat_rate']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def calculate_trifecta_probs(win_probs):
    combos = list(itertools.permutations(range(1, 7), 3))
    trifecta_probs = {}
    for c in combos:
        p1 = win_probs[c[0]]
        p2 = win_probs[c[1]] / (1 - p1 + 1e-9)
        p3 = win_probs[c[2]] / (1 - p1 - win_probs[c[1]] + 1e-9)
        trifecta_probs[f"{c[0]}{c[1]}{c[2]}"] = max(0, p1 * p2 * p3)
    total = sum(trifecta_probs.values())
    return {k: v / total for k, v in trifecta_probs.items()}

def run_evaluation():
    # Load Models
    models = {}
    models['lgb'] = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'lgb_model.txt'))
    models['cb'] = cb.CatBoostClassifier().load_model(os.path.join(MODEL_DIR, 'cb_model.bin'))
    with open(os.path.join(MODEL_DIR, 'alt_model.pkl'), 'rb') as f:
        models['alt'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'alt_scaler.pkl'), 'rb') as f:
        models['alt_scaler'] = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'rb') as f:
        models['meta'] = pickle.load(f)

    # Load Data (Validation Set part)
    df = load_data(2000) # Latest 2000 races
    features = ['venue_code', 'exhibition_time', 'exhibition_start_timing', 'pred_course', 
                'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_id']
    
    # Inference
    X = df[features]
    lgb_probs = models['lgb'].predict(X)
    cb_probs = models['cb'].predict_proba(X)[:, 1]
    X_alt_sub = X.drop(['venue_code', 'racer_id'], axis=1)
    X_alt_scaled = models['alt_scaler'].transform(X_alt_sub)
    alt_probs = models['alt'].predict_proba(X_alt_scaled)[:, 1]
    
    X_meta = np.vstack([lgb_probs, cb_probs, alt_probs]).T
    df['pred_win_prob'] = models['meta'].predict_proba(X_meta)[:, 1]
    
    # Group by race
    groups = df.groupby('race_id')
    
    conn = sqlite3.connect(DB_PATH)
    
    total_bet = 0
    total_return = 0
    hits = 0
    betted_races = 0
    
    ev_threshold = 5.0 # High sensitivity threshold
    
    print(f"Starting simulation on {len(groups)} races...")
    
    for i, (rid, group) in enumerate(groups):
        if i % 100 == 0: print(f"Processing... {i}/{len(groups)}")
        
        # Win dict
        win_dict = dict(zip(group['boat_number'], group['pred_win_prob']))
        # Normalize
        s = sum(win_dict.values()) + 1e-9
        win_dict = {k: v/s for k, v in win_dict.items()}
        
        # Results
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            winning_combo = f"{r1}{r2}{r3}"
        except:
            continue
            
        # Get Odds from DB
        odds_df = pd.read_sql(f"SELECT combination, odds_1min FROM odds_data WHERE race_id = '{rid}'", conn)
        odds_dict = dict(zip(odds_df['combination'], odds_df['odds_1min']))
        
        if not odds_dict: continue
        
        # Trifecta Probs
        trifecta_probs = calculate_trifecta_probs(win_dict)
        
        # Check EV
        race_betted = False
        for combo, prob in trifecta_probs.items():
            odds = odds_dict.get(combo, 0)
            if odds <= 0: continue
            
            ev = prob * odds
            if ev >= ev_threshold:
                total_bet += 100
                race_betted = True
                if combo == winning_combo:
                    total_return += 100 * odds
                    hits += 1
        
        if race_betted: betted_races += 1

    conn.close()
    
    recovery_rate = (total_return / total_bet * 100) if total_bet > 0 else 0
    
    print("\n=== ROI Evaluation Results (EV Threshold: 2.0) ===")
    print(f"Total Evaluated Races: {len(groups)}")
    print(f"Betted Races: {betted_races}")
    print(f"Total Bet Amount: {total_bet} JPY")
    print(f"Total Return Amount: {int(total_return)} JPY")
    print(f"Recovery Rate (ROI): {recovery_rate:.2f}%")
    print(f"Hit Count: {hits}")
    print(f"Hit Rate (Betted): {hits/total_bet*100*100 if total_bet>0 else 0:.2f}% (Average combo count per race: {total_bet/100/betted_races if betted_races>0 else 0:.1f})")

if __name__ == "__main__":
    run_evaluation()
