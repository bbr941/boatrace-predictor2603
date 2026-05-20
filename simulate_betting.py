import pandas as pd
import numpy as np
import lightgbm as lgb
import sqlite3
import itertools
import os
import train_model
import math

# Config
MODEL_PATH = 'model_honmei.txt'
MODEL_ANA_PATH = 'model_ana.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'

# ==========================================
# 運用モード設定
# ==========================================
USE_PLAN_B = False  # False: Plan A (全レース参戦), True: Plan B (厳選フィルター)

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def calculate_plackett_luce_probs(honmei_scores_dict):
    """
    本命スコアからSoftmaxとPlackett-Luceモデルを用いて3連単の合成確率を算出する
    Returns: pl_probs (list), max_p1 (float)
    """
    boats = list(honmei_scores_dict.keys())
    scores = np.array([honmei_scores_dict[b] for b in boats])
    
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    p1 = exp_scores / np.sum(exp_scores)
    max_p1 = float(np.max(p1))
    
    p1_dict = {boats[i]: p1[i] for i in range(len(boats))}
    
    combos = list(itertools.permutations(boats, 3))
    pl_probs = []
    
    for c in combos:
        b1, b2, b3 = c
        prob1 = p1_dict[b1]
        denom2 = 1.0 - p1_dict[b1]
        if denom2 <= 0: denom2 = 1e-9
        prob2 = p1_dict[b2] / denom2
        denom3 = 1.0 - p1_dict[b1] - p1_dict[b2]
        if denom3 <= 0: denom3 = 1e-9
        prob3 = p1_dict[b3] / denom3
        
        total_prob = prob1 * prob2 * prob3
        combo_str = f"{b1}-{b2}-{b3}"
        pl_probs.append({'combo': combo_str, 'prob': total_prob})
        
    pl_probs.sort(key=lambda x: x['prob'], reverse=True)
    return pl_probs, max_p1

def get_all_trifecta_odds(conn, race_id):
    query = "SELECT combination, odds_1min FROM odds_data WHERE race_id = ? AND length(combination) = 3"
    try:
        cursor = conn.cursor()
        cursor.execute(query, [race_id])
        rows = cursor.fetchall()
        
        odds_map = {}
        for r in rows:
            comb_db = str(r[0])
            if len(comb_db) == 3:
                val = r[1]
                comb_fmt = f"{comb_db[0]}-{comb_db[1]}-{comb_db[2]}"
                odds_map[comb_fmt] = val
        return odds_map
    except Exception as e:
        return {}

def select_hybrid_formation(pl_probs, ana_scores_dict, all_odds):
    if not pl_probs or not all_odds:
        return []
        
    top_combo = pl_probs[0]['combo']
    top_odds = all_odds.get(top_combo, 0)
    
    if top_odds < 1:
        return [] 
        
    N = int(min(8, math.floor(top_odds)))
    if N < 1:
        return [] 
        
    selected = [p['combo'] for p in pl_probs[:N]]
    
    best_ana_boat = max(ana_scores_dict, key=ana_scores_dict.get)
    best_ana_boat_str = str(best_ana_boat)
    
    ana_in_3rd = any(c.split('-')[2] == best_ana_boat_str for c in selected)
    
    if not ana_in_3rd:
        parts = top_combo.split('-')
        b1, b2 = parts[0], parts[1]
        if best_ana_boat_str != b1 and best_ana_boat_str != b2:
            new_combo = f"{b1}-{b2}-{best_ana_boat_str}"
            if new_combo not in selected:
                selected[-1] = new_combo
                
    return selected

def calculate_funds_distribution(selected_combos, pl_probs_list, all_odds, base_return=1000, bonus_budget=500):
    if not selected_combos:
        return {}
        
    pl_probs_dict = {p['combo']: p['prob'] for p in pl_probs_list}
    
    for c in selected_combos:
        if all_odds.get(c, 0) < 1.01:
            return {} 
            
    sum_p = sum(pl_probs_dict.get(c, 0) for c in selected_combos)
    if sum_p <= 0: sum_p = 1.0
    
    bets = {}
    for c in selected_combos:
        o = all_odds[c]
        p = pl_probs_dict.get(c, 0)
        
        b_base = base_return / o
        b_bonus = bonus_budget * (p / sum_p)
        
        raw_bet = b_base + b_bonus
        bet_100 = max(100, round(raw_bet / 100) * 100)
        bets[c] = bet_100
        
    total_bet = sum(bets.values())
    for c, b in bets.items():
        if b * all_odds[c] <= total_bet:
            bets_flat = {c: 100 for c in selected_combos}
            total_flat = len(selected_combos) * 100
            for cf, bf in bets_flat.items():
                if bf * all_odds[cf] <= total_flat:
                    return {} 
            return bets_flat 
            
    return bets

def run_simulation():
    print("Loading Data...")
    df = pd.read_csv(DATA_PATH)
    
    df = train_model.preprocess_data(df)
    
    print("Fetching valid Race IDs from DB...")
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT 10")
    valid_races = [row[0] for row in cursor.fetchall()]
    
    test_df = df[df['race_id'].isin(valid_races)].copy()
    test_races = test_df['race_id'].unique()
    print(f"Test Set: {len(test_races)} races")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MODEL_ANA_PATH):
        print("Models not found.")
        return

    print("Loading Models...")
    model_honmei = lgb.Booster(model_file=MODEL_PATH)
    model_ana = lgb.Booster(model_file=MODEL_ANA_PATH)
    
    feats_honmei = model_honmei.feature_name()
    for f in feats_honmei:
        if f not in test_df.columns: test_df[f] = 0
    print("Predicting Honmei...")
    test_df['score_honmei'] = model_honmei.predict(test_df[feats_honmei])
    
    feats_ana = model_ana.feature_name()
    for f in feats_ana:
        if f not in test_df.columns: test_df[f] = 0
    print("Predicting Ana...")
    test_df['score_ana'] = model_ana.predict(test_df[feats_ana])
    
    race_results_cache = []
    groups = test_df.groupby('race_id')
    
    count = 0
    for rid, group in groups:
        count += 1
        if count % 100 == 0:
            print(f"Processing calculations {count}/{len(test_races)}...", end='\r')
            
        honmei_scores = dict(zip(group['boat_number'], group['score_honmei']))
        ana_scores = dict(zip(group['boat_number'], group['score_ana']))
        
        pl_probs, max_p1 = calculate_plackett_luce_probs(honmei_scores)
        all_odds = get_all_trifecta_odds(conn, rid)
        
        prob_gap = 0.0
        if len(pl_probs) >= 2:
            prob_gap = pl_probs[0]['prob'] - pl_probs[1]['prob']
        
        selected_combos = select_hybrid_formation(pl_probs, ana_scores, all_odds)
        bets = calculate_funds_distribution(selected_combos, pl_probs, all_odds)
        
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            actual_combo = f"{int(r1)}-{int(r2)}-{int(r3)}"
        except IndexError:
            actual_combo = None
            
        race_results_cache.append({
            'race_id': rid,
            'max_p1': max_p1,
            'prob_gap': prob_gap,
            'bets': bets,
            'all_odds': all_odds,
            'actual_combo': actual_combo
        })
        
    conn.close()
    print("\nCalculations finished.")
    
    total_races = len(race_results_cache)
    
    if not USE_PLAN_B:
        stats = {'betted': 0, 'hits': 0, 'bet_amt': 0, 'return_amt': 0}
        for r in race_results_cache:
            if r['bets']:
                stats['betted'] += 1
                stats['bet_amt'] += sum(r['bets'].values())
                if r['actual_combo'] and r['actual_combo'] in r['bets']:
                    stats['hits'] += 1
                    stats['return_amt'] += r['bets'][r['actual_combo']] * r['all_odds'].get(r['actual_combo'], 0)
        
        print("\n\n=== Simulation Results (Plan A) ===")
        print(f"Total Races processed: {total_races}")
        print(f"Betted Races: {stats['betted']} ({stats['betted']/total_races:.1%})")
        if stats['betted'] > 0:
            print(f"Hit Rate (When Betted): {stats['hits'] / stats['betted']:.2%}")
            print(f"Recovery Rate (ROI): {stats['return_amt'] / stats['bet_amt']:.2%}")
            print(f"Total Profit: {int(stats['return_amt'] - stats['bet_amt'])} JPY")
            
    else:
        # P1 >= 0.49, Gap >= 0.010 (Optimal params from grid search)
        p1_th = 0.49
        gap_th = 0.010
        
        stats = {'betted': 0, 'hits': 0, 'bet_amt': 0, 'return_amt': 0}
        for r in race_results_cache:
            if r['max_p1'] >= p1_th and r['prob_gap'] >= gap_th:
                bets = r['bets']
                if not bets: continue 
                
                stats['betted'] += 1
                stats['bet_amt'] += sum(bets.values())
                
                if r['actual_combo'] and r['actual_combo'] in bets:
                    stats['hits'] += 1
                    stats['return_amt'] += bets[r['actual_combo']] * r['all_odds'].get(r['actual_combo'], 0)
                    
        betted_rate = stats['betted'] / total_races
        roi = stats['return_amt'] / stats['bet_amt'] if stats['bet_amt'] > 0 else 0
        
        print("\n\n=== Simulation Results (Plan B - Best Params) ===")
        print(f"Total Races processed: {total_races}")
        print(f"Betted Races: {stats['betted']} ({betted_rate:.1%})")
        print(f"Hit Rate (When Betted): {stats['hits'] / stats['betted']:.2%}")
        print(f"Hit Rate (Global): {stats['hits'] / total_races:.2%}")
        print(f"Recovery Rate (ROI): {roi:.2%}")
        print(f"Avg Bet Amount per Race: {int(stats['bet_amt'] / stats['betted'])} JPY")
        print(f"Total Profit: {int(stats['return_amt'] - stats['bet_amt'])} JPY")

if __name__ == "__main__":
    run_simulation()
