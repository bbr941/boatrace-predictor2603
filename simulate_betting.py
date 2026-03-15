
import pandas as pd
import numpy as np
import lightgbm as lgb
import sqlite3
import itertools
import os
import train_model

# Config
MODEL_PATH = 'model_honmei.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'

# Constants
INITIAL_BET = 100
BET_STEP = 100
MAX_BET_PER_BOAT = 2000
MAX_TOTAL_BET = 5000

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def calculate_trifecta_scores(scores, sorted_boats):
    """
    Generate Top 5 Trifecta Combinations based on individual boat scores.
    """
    # Simple logic: Top 5 scored boats permutation?
    # Or just permutations of top N boats sorted by sum of scores?
    # app_boatrace uses: 
    # combs = list(itertools.permutations(sorted_boats[:4], 3)) 
    # score = sum(scores[b] for b in comb)
    # Then sort.
    
    # We replicate app logic
    # Take top 4 boats to form combos (4P3 = 24 combos)
    # Filter top 5 by combined score
    
    candidates = sorted_boats[:4]
    combos = list(itertools.permutations(candidates, 3))
    
    combo_data = []
    for c in combos:
        # Score sum
        s = sum(scores[b] for b in c)
        combo_str = f"{c[0]}-{c[1]}-{c[2]}"
        combo_data.append({'combo': combo_str, 'score': s})
        
    df_combos = pd.DataFrame(combo_data)
    df_combos = df_combos.sort_values('score', ascending=False).head(5)
    return df_combos

def get_odds_from_db(conn, race_id, combos):
    """
    Fetch trifecta odds for specific combinations from DB.
    """
    # Format: combination '1-2-3' matches DB '123' if '123' layout.
    # DB has '123'. app_boatrace uses '1-2-3'.
    # We must strip hyphens.
    
    target_combinations = [c.replace('-', '') for c in combos]
    
    if not target_combinations:
        return {}
        
    placeholders = ",".join(["?"] * len(target_combinations))
    query = f"""
        SELECT combination, odds_1min 
        FROM odds_data 
        WHERE race_id = ? AND combination IN ({placeholders})
    """
    
    params = [race_id] + target_combinations
    try:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Map back to '1-2-3' format if needed or keep raw
        # Return dict: {'1-2-3': 15.5, ...}
        
        odds_map = {}
        for r in rows:
            comb_db = r[0] # '123'
            val = r[1]
            # Convert '123' -> '1-2-3'
            comb_fmt = f"{comb_db[0]}-{comb_db[1]}-{comb_db[2]}"
            odds_map[comb_fmt] = val
            
        return odds_map
    except Exception as e:
        # print(f"Error fetching odds: {e}")
        return {}

def staircase_betting_logic(predictions_df):
    """
    predictions_df: DataFrame with columns ['combo', 'odds', 'score']
    Returns: list of bet amounts corresponding to rows
    """
    # Sort by Odds Ascending
    # predictions_df must have 'odds' column populated. Nan if missing.
    
    df = predictions_df.copy()
    df['bet'] = 0
    
    # Filter valid odds
    df = df.dropna(subset=['odds'])
    df = df[df['odds'] > 0]
    df = df.sort_values('odds', ascending=True)
    
    if df.empty:
        return {} # No bets
        
    total_investment = 0
    bets = {} # combo -> amount
    
    for idx, row in df.iterrows():
        combo = row['combo']
        odds = row['odds']
        
        # Determine bet amount
        # Condition: (current_bet * odds) > current_total_investment + current_bet
        # i.e., Profit > 0
        # Wait, if I increase bet, total_investment increases.
        # Break-even: Revenue >= Cost
        # Revenue = bet * odds
        # Cost = total_investment_so_far + bet
        # So: bet * odds > total_investment_so_far + bet
        # bet * (odds - 1) > total_investment_so_far
        # bet > total_investment_so_far / (odds - 1)
        
        # Start from 100
        bet = 100
        
        # Adjust for first bet?
        # If total_invest is 0. bet > 0. 100 is fine.
        
        # Loop until profit
        while True:
            cost = total_investment + bet
            revenue = bet * odds
            if revenue > cost:
                break
            bet += 100
            
            # Constraints
            if bet > MAX_BET_PER_BOAT:
                bet = 0 # Give up on this combo
                break
            if (total_investment + bet) > MAX_TOTAL_BET:
                bet = 0 # Give up
                break
                
        if bet > 0:
            bets[combo] = bet
            total_investment += bet
        else:
            # Skip this combo (0 bet)
            pass
            
    return bets

def run_simulation():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocess
    df = train_model.preprocess_data(df)
    
    # Select races that exist in DB odds_data to ensure valid simulation
    print("Fetching valid Race IDs from DB...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT 5000")
    valid_races = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    test_df = df[df['race_id'].isin(valid_races)].copy()
    test_races = test_df['race_id'].unique()
    
    print(f"Test Set: {len(test_races)} races")
    
    if not os.path.exists(MODEL_PATH):
        print("Model not found")
        return

    print("Loading Model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    feats = model.feature_name()
    
    for f in feats:
        if f not in test_df.columns:
            test_df[f] = 0
            
    print("Predicting...")
    preds = model.predict(test_df[feats])
    test_df['score'] = preds
    
    # DB Connect
    conn = get_db_connection()
    
    # Simulation Stats
    stats = {
        'total_races': 0,
        'betted_races': 0,
        'hits': 0,
        'total_bet': 0,
        'total_return': 0
    }
    
    # Iterate Races
    # Group by race_id
    groups = test_df.groupby('race_id')
    
    count = 0
    for rid, group in groups:
        count += 1
        if count % 100 == 0:
            print(f"Propcessing {count}/{len(test_races)}...", end='\r')
            
        # 1. Get Top 5 Combos
        scores = dict(zip(group['boat_number'], group['score']))
        sorted_boats = group.sort_values('score', ascending=False)['boat_number'].tolist()
        
        df_combos = calculate_trifecta_scores(scores, sorted_boats)
        top_combos = df_combos['combo'].tolist() # ['1-2-3', ...]
        
        # 2. Get Real Odds
        odds_map = get_odds_from_db(conn, rid, top_combos)
        
        if count < 5:
             print(f"\nDEBUG Race {rid}:")
             print(f"  Top Combos: {top_combos}")
             print(f"  Odds Map: {odds_map}")
        
        # Add odds to df_combos
        df_combos['odds'] = df_combos['combo'].map(odds_map)
        
        # 3. Betting Logic
        bets = staircase_betting_logic(df_combos)
        
        if not bets:
            stats['total_races'] += 1
            continue
            
        stats['total_races'] += 1
        stats['betted_races'] += 1
        
        race_bet = sum(bets.values())
        race_return = 0
        
        stats['total_bet'] += race_bet
        
        # 4. Check Result
        # We need the Actual Result Trifecta
        # Using `rank` column in group.
        # Find 1st, 2nd, 3rd boats
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            actual_combo = f"{r1}-{r2}-{r3}"
            
            if actual_combo in bets:
                stats['hits'] += 1
                hit_amount = bets[actual_combo]
                hit_odds = odds_map.get(actual_combo, 0)
                race_return = hit_amount * hit_odds
                
        except IndexError:
            # Missing rank info?
            pass
            
        stats['total_return'] += race_return
        
    conn.close()
    
    print("\n\n=== Simulation Results (Staircase Break-Even) ===")
    print(f"Total Races processed: {stats['total_races']}")
    print(f"Betted Races: {stats['betted_races']} ({stats['betted_races']/stats['total_races']:.1%})")
    
    if stats['betted_races'] > 0:
        avg_hit_rate = stats['hits'] / stats['betted_races'] # Among betted
        global_hit_rate = stats['hits'] / stats['total_races'] # Among all
        recovery_rate = stats['total_return'] / stats['total_bet'] if stats['total_bet'] > 0 else 0
        avg_bet_amount = stats['total_bet'] / stats['betted_races']
        
        print(f"Hit Rate (When Betted): {avg_hit_rate:.2%}")
        print(f"Hit Rate (Global): {global_hit_rate:.2%}")
        print(f"Recovery Rate: {recovery_rate:.2%}")
        print(f"Avg Bet Amount: {int(avg_bet_amount)} JPY")
        print(f"Total Profit: {int(stats['total_return'] - stats['total_bet'])} JPY")
    else:
        print("No bets made.")

if __name__ == "__main__":
    run_simulation()
