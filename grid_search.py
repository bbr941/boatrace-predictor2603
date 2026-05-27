import pandas as pd
import numpy as np
import sqlite3
import lightgbm as lgb
from simulate_betting import calculate_plackett_luce_probs, select_hybrid_formation, calculate_funds_distribution

def sweep_parameters():
    # 1. Load Data
    print("Loading Data...")
    df = pd.read_csv('boatrace_dataset_labeled_v2.csv')
    
    # Simple preprocessing from train_model (fill na, etc.)
    import train_model
    df = train_model.preprocess_data(df)
    
    # 2. Get latest 5000 races
    print("Fetching valid Race IDs...")
    conn = sqlite3.connect(r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT 5000")
    valid_races = [row[0] for row in cursor.fetchall()]
    
    test_df = df[df['race_id'].isin(valid_races)].copy()
    test_races = test_df['race_id'].unique().tolist()
    print(f"Test Set: {len(test_races)} races")
    
    # 3. Predict Models
    print("Loading Models and Predicting...")
    model_honmei = lgb.Booster(model_file='model_honmei.txt')
    model_ana = lgb.Booster(model_file='model_ana.txt')
    
    feats_honmei = model_honmei.feature_name()
    for f in feats_honmei:
        if f not in test_df.columns: test_df[f] = 0
    test_df['score_honmei'] = model_honmei.predict(test_df[feats_honmei])
    
    feats_ana = model_ana.feature_name()
    for f in feats_ana:
        if f not in test_df.columns: test_df[f] = 0
    test_df['score_ana'] = model_ana.predict(test_df[feats_ana])
    
    # 4. Load all odds for 5000 races at once to save time
    print("Fetching all odds from DB at once...")
    chunk_size = 500
    all_odds_dict = {}
    for i in range(0, len(test_races), chunk_size):
        chunk = test_races[i:i+chunk_size]
        placeholders = ','.join(['?']*len(chunk))
        q = f"SELECT race_id, combination, odds_1min FROM odds_data WHERE race_id IN ({placeholders}) AND length(combination) = 3"
        cursor.execute(q, chunk)
        for r in cursor.fetchall():
            rid = r[0]
            comb_db = str(r[1])
            val = r[2]
            comb_fmt = f"{comb_db[0]}-{comb_db[1]}-{comb_db[2]}"
            if rid not in all_odds_dict:
                all_odds_dict[rid] = {}
            all_odds_dict[rid][comb_fmt] = val

    conn.close()
    
    # 5. Simulate each race once
    print("Simulating bets...")
    race_results_cache = []
    groups = test_df.groupby('race_id')
    
    for rid, group in groups:
        honmei_scores = dict(zip(group['boat_number'], group['score_honmei']))
        ana_scores = dict(zip(group['boat_number'], group['score_ana']))
        
        pl_probs, max_p1 = calculate_plackett_luce_probs(honmei_scores)
        all_odds = all_odds_dict.get(rid, {})
        
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
            'max_p1': max_p1,
            'prob_gap': prob_gap,
            'bets': bets,
            'all_odds': all_odds,
            'actual_combo': actual_combo
        })
        
    print("Sweeping parameters...")
    
    p1_thresholds = [0.0, 0.45, 0.47, 0.49, 0.51]
    gap_thresholds = [0.0, 0.005, 0.010, 0.015]
    
    total_races = len(race_results_cache)
    
    # Generate Markdown Table
    out = []
    out.append("| P1閾値 | Gap閾値 | 対象レース数 | 参加率 | 的中率 | 回収率 (ROI) | 利益 (円) |")
    out.append("|---|---|---|---|---|---|---|")
    
    for p1 in p1_thresholds:
        for gap in gap_thresholds:
            stats = {'betted': 0, 'hits': 0, 'bet_amt': 0, 'return_amt': 0}
            for r in race_results_cache:
                if r['max_p1'] >= p1 and r['prob_gap'] >= gap:
                    if r['bets']:
                        stats['betted'] += 1
                        stats['bet_amt'] += sum(r['bets'].values())
                        if r['actual_combo'] and r['actual_combo'] in r['bets']:
                            stats['hits'] += 1
                            stats['return_amt'] += r['bets'][r['actual_combo']] * r['all_odds'].get(r['actual_combo'], 0)
            
            if stats['betted'] > 0:
                hit_rate = stats['hits'] / stats['betted']
                roi = stats['return_amt'] / stats['bet_amt']
                profit = int(stats['return_amt'] - stats['bet_amt'])
                part_rate = stats['betted'] / total_races
                out.append(f"| {p1:.2f} | {gap:.3f} | {stats['betted']} | {part_rate:.1%} | {hit_rate:.1%} | {roi:.1%} | {profit:,} |")
            else:
                out.append(f"| {p1:.2f} | {gap:.3f} | 0 | 0.0% | - | - | 0 |")
                
    with open('sweep_results.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(out))
    
    print("Saved to sweep_results.md")

if __name__ == '__main__':
    sweep_parameters()
