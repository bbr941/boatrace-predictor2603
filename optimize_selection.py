
import pandas as pd
import numpy as np
import lightgbm as lgb
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
import train_model

# Config
MODEL_PATH = 'model_honmei.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def analyze_features():
    print("--- Step 1: Feature Importance Analysis ---")
    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    model = lgb.Booster(model_file=MODEL_PATH)
    importance = model.feature_importance(importance_type='gain')
    feature_names = model.feature_name()
    
    # Create DataFrame
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df_imp = df_imp.sort_values('importance', ascending=False)
    
    print("\nTop 20 Features:")
    print(df_imp.head(20))
    
    print("\nBottom 20 Features (Candidates for Removal):")
    print(df_imp.tail(20))
    
    return df_imp

def calculate_confidence_metrics(df):
    """
    Calculate confidence metrics per race based on 'score'.
    Assumes df has 'race_id' and 'score'.
    """
    # Group by race
    # Calculate Std and Gap (1st - 2nd)
    
    stats = df.groupby('race_id')['score'].agg(['std', 'max', lambda x: x.nlargest(2).iloc[-1] if len(x) >= 2 else 0]).reset_index()
    stats.columns = ['race_id', 'score_std', 'score_max', 'score_2nd']
    stats['score_gap'] = stats['score_max'] - stats['score_2nd']
    
    return stats[['race_id', 'score_std', 'score_gap']]

def get_trifecta_odds(conn, race_ids):
    # Fetch all odds for test races efficiently? 
    # Or fetch on demand? On demand is slower but simpler given the schema.
    # To speed up, we can fetch all for these races if possible.
    # But let's reuse the fetch-on-demand logic for simplicity or optimize if needed.
    # Actually, fetching *all* odds for 5000 races is huge (120 combos * 5000 = 600k rows).
    # Memory wise it's ~50MB. Doable.
    
    print("Fetching Odds for all test races (Batch)...")
    placeholders = ",".join(["?"] * len(race_ids))
    # Optimize: Fetch only top combos? No, we don't know top combos yet.
    # We'll fetch on demand inside the loop, same as simulate_betting.py
    # But we can optimize to fetch *all* combos for a single race in one query.
    return None



def run_grid_search_group_b():
    print("\n--- Step 5: Grid Search (Group B Optimization) ---")
    
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    df = train_model.preprocess_data(df)
    
    # 2. Filter Valid Races
    print("Filtering Valid Races...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT 5000")
    valid_races_db = {row[0] for row in cursor.fetchall()}
    
    df = df[df['race_id'].isin(valid_races_db)].copy()
    
    # 3. Predict
    print("Predicting Scores...")
    model = lgb.Booster(model_file=MODEL_PATH)
    feats = model.feature_name()
    for f in feats:
        if f not in df.columns: df[f] = 0
    df['score'] = model.predict(df[feats])
    
    # 4. Calculate Confidence
    print("Calculating Confidence Metrics...")
    conf_df = calculate_confidence_metrics(df) # race_id, score_std
    conf_df = conf_df.sort_values('score_std', ascending=False)
    
    total_races = len(conf_df)
    races_unique = df['race_id'].unique()
    df_grouped = df.groupby('race_id')
    
    # Pre-fetch odds to speed up grid search? 
    # With 5000 races * 9 iterations, fetching odds 9 times is slow.
    # We should iterate races ONCE, fetch odds, and then check against all 9 conditions?
    # Or just cache odds in memory.
    # There are 5000 races. Each has ~20-120 odds.
    # Let's fetch Top 6 combos' odds for each race and store them.
    # Top 6 candidates -> Permutations(6,3)=120 combos? No, betting Top 6 *bets*.
    # Candidates: Top 4 boat numbers form 24 combos. The Top 6 bets will be among them.
    # So we need odds for the Top 24 combos (just to be safe) or verify Top 6 bets are within.
    
    # Let's cache the "race info object" with:
    # - race_id
    # - Top 6 Combos (Strings) + Scores
    # - Odds for those Top 6 (fetched from DB)
    # - Actual Result Combo
    
    print("Caching Race Data & Odds (Top 6 Picks)...")
    race_cache = []
    
    count = 0
    for rid in races_unique:
        count += 1
        if count % 500 == 0: print(f"Caching {count}/{total_races}...", end='\r')
        
        group = df_grouped.get_group(rid)
        
        # Candidates (Standardize on Top 4 boats for permutation base)
        # Assuming Top 6 bets usually come from Top 4 boats.
        candidates = group.sort_values('score', ascending=False)['boat_number'].tolist()[:4]
        combos = list(itertools.permutations(candidates, 3))
        
        scores = dict(zip(group['boat_number'], group['score']))
        combo_data = []
        for c in combos:
            s = sum(scores[b] for b in c)
            combo_data.append({'combo_str': f"{c[0]}-{c[1]}-{c[2]}", 'score': s})
            
        # Top 6 Bets (for the max strategy)
        targets = sorted(combo_data, key=lambda x: x['score'], reverse=True)[:6]
        
        # Fetch Odds
        req_combos_str = [x['combo_str'].replace('-', '') for x in targets]
        p_holder = ",".join(["?"] * len(req_combos_str))
        q = f"SELECT combination, odds_1min FROM odds_data WHERE race_id = ? AND combination IN ({p_holder})"
        cursor = conn.cursor()
        cursor.execute(q, [rid] + req_combos_str)
        rows_odds = cursor.fetchall()
        
        odds_map = {}
        for r_o in rows_odds:
            c_db = r_o[0]
            c_fmt = f"{c_db[0]}-{c_db[1]}-{c_db[2]}"
            odds_map[c_fmt] = r_o[1]
            
        # Actual
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            actual = f"{r1}-{r2}-{r3}"
        except: actual = None
        
        race_cache.append({
            'race_id': rid,
            'targets': targets, # List of dicts {'combo_str', 'score'}
            'odds_map': odds_map,
            'actual': actual
        })
        
    conn.close()
    print("\nCache Complete.")
    
    # 5. Grid Search
    # Depths (Bottom %): 0.5, 0.3, 0.2
    # Odds Floors: 15.0, 20.0, 30.0
    
    depths = [0.5, 0.3, 0.2]
    odds_floors = [15.0, 20.0, 30.0]
    
    print("\n=== Grid Search Results (Group B) ===")
    print(f"{'Depth':<10} | {'Odds >=':<8} | {'Races':<6} | {'Hit%':<6} | {'Recov%':<7} | {'Profit':<8}")
    
    for depth in depths:
        # Filter Races (Bottom X%)
        # conf_df is sorted Descending.
        # Bottom X% starts at index: total * (1 - depth)
        start_idx = int(total_races * (1 - depth))
        target_races_ids = set(conf_df.iloc[start_idx:]['race_id'])
        
        for floor in odds_floors:
            
            stats = {'races': 0, 'betted': 0, 'hits': 0, 'bet': 0, 'ret': 0}
            
            for r_data in race_cache:
                rid = r_data['race_id']
                if rid not in target_races_ids: continue
                
                stats['races'] += 1
                
                bet_amt = 0
                ret_amt = 0
                has_bet = False
                
                # Check bets
                for t in r_data['targets']: # Top 6
                    c_str = t['combo_str']
                    odds = r_data['odds_map'].get(c_str, 0.0)
                    
                    if odds >= floor:
                        bet_amt += 100
                        has_bet = True
                        if c_str == r_data['actual']:
                            ret_amt += (100 * odds)
                            
                if has_bet:
                    stats['betted'] += 1
                    stats['bet'] += bet_amt
                    stats['ret'] += ret_amt
                    if ret_amt > 0:
                        stats['hits'] += 1
                        
            # Metrics
            hit_rate = (stats['hits'] / stats['betted'] * 100) if stats['betted'] > 0 else 0
            recov = (stats['ret'] / stats['bet'] * 100) if stats['bet'] > 0 else 0
            profit = stats['ret'] - stats['bet']
            
            label_depth = f"Bot {int(depth*100)}%"
            print(f"{label_depth:<10} | {floor:<8} | {stats['betted']:<6} | {hit_rate:.1f}%  | {recov:.1f}%   | {int(profit):<8}")

if __name__ == "__main__":
    import itertools
    # analyze_features() 
    run_grid_search_group_b()
