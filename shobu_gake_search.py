import pandas as pd
import numpy as np
import sqlite3
import lightgbm as lgb
from simulate_betting import calculate_plackett_luce_probs, select_hybrid_formation, calculate_funds_distribution
import json

def generate_shobu_gake_report():
    print("Loading Data...")
    df = pd.read_csv('boatrace_dataset_labeled_v2.csv')
    
    import train_model
    df = train_model.preprocess_data(df)
    
    print("Fetching valid Race IDs...")
    conn = sqlite3.connect(r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db')
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT 5000")
    valid_races = [row[0] for row in cursor.fetchall()]
    
    test_df = df[df['race_id'].isin(valid_races)].copy()
    test_races = test_df['race_id'].unique().tolist()
    
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
    
    print("Fetching all odds from DB at once...")
    chunk_size = 500
    all_odds_dict = {}
    for i in range(0, len(test_races), chunk_size):
        chunk = test_races[i:i+chunk_size]
        placeholders = ','.join(['?']*len(chunk))
        q = f"SELECT race_id, combination, odds_1min FROM odds_data WHERE race_id IN ({placeholders}) AND length(combination) = 3"
        cursor.execute(q, chunk)
        for r in cursor.fetchall():
            rid, comb_db, val = r[0], str(r[1]), r[2]
            comb_fmt = f"{comb_db[0]}-{comb_db[1]}-{comb_db[2]}"
            if rid not in all_odds_dict: all_odds_dict[rid] = {}
            all_odds_dict[rid][comb_fmt] = val
    conn.close()
    
    print("Simulating base bets...")
    results = []
    groups = test_df.groupby('race_id')
    
    for rid, group in groups:
        honmei_scores = dict(zip(group['boat_number'], group['score_honmei']))
        ana_scores = dict(zip(group['boat_number'], group['score_ana']))
        
        pl_probs, max_p1 = calculate_plackett_luce_probs(honmei_scores)
        all_odds = all_odds_dict.get(rid, {})
        
        prob_gap = pl_probs[0]['prob'] - pl_probs[1]['prob'] if len(pl_probs) >= 2 else 0.0
        
        selected_combos = select_hybrid_formation(pl_probs, ana_scores, all_odds)
        bets = calculate_funds_distribution(selected_combos, pl_probs, all_odds)
        
        if not bets:
            continue
            
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            actual_combo = f"{int(r1)}-{int(r2)}-{int(r3)}"
        except IndexError:
            actual_combo = None
            
        # Metrics for triggering Shobu-gake
        top_combo = pl_probs[0]['combo']
        top_odds = all_odds.get(top_combo, 0)
        ev_top = pl_probs[0]['prob'] * top_odds
        
        ev_sum = 0
        pl_probs_dict = {p['combo']: p['prob'] for p in pl_probs}
        for c in selected_combos:
            ev_sum += pl_probs_dict.get(c, 0) * all_odds.get(c, 0)
            
        # Ana score for the race
        max_ana_score = max(ana_scores.values())
        
        base_bet_amt = sum(bets.values())
        base_return_amt = 0
        if actual_combo and actual_combo in bets:
            base_return_amt = bets[actual_combo] * all_odds.get(actual_combo, 0)
            
        results.append({
            'race_id': rid,
            'max_p1': max_p1,
            'prob_gap': prob_gap,
            'ev_top': ev_top,
            'ev_sum': ev_sum,
            'max_ana_score': max_ana_score,
            'base_bet_amt': base_bet_amt,
            'base_return_amt': base_return_amt,
            'hit': 1 if base_return_amt > 0 else 0
        })
        
    res_df = pd.DataFrame(results)
    total_betted = len(res_df)
    target_count = int(total_betted * 0.20)
    
    print(f"Total Betted Races: {total_betted}")
    print(f"Target 20% Races: {target_count}")
    
    metrics = [
        ('max_p1', True), # higher is better
        ('prob_gap', True),
        ('ev_top', True),
        ('ev_sum', True),
        ('max_ana_score', True)
    ]
    
    base_profit = res_df['base_return_amt'].sum() - res_df['base_bet_amt'].sum()
    base_roi = res_df['base_return_amt'].sum() / res_df['base_bet_amt'].sum()
    print(f"Base ROI: {base_roi:.2%}, Base Profit: {base_profit:,.0f}")
    
    report = []
    report.append("# 勝負掛けモード 条件探索レポート\n")
    report.append(f"**通常時（全体）**\n- 購入レース数: {total_betted}\n- 回収率: {base_roi:.2%}\n- 利益: {base_profit:,.0f}円\n")
    
    multipliers = [2, 3, 4, 5]
    
    best_strategy = None
    best_profit = -float('inf')
    
    for metric, asc in metrics:
        report.append(f"## トリガー候補: `{metric}` (上位20%)")
        report.append("| 倍率 | 勝負掛け件数 | 勝負対象のROI | 全体回収率 | 全体利益 | 利益差分 |")
        report.append("|---|---|---|---|---|---|")
        
        # Sort and get threshold
        sorted_df = res_df.sort_values(by=metric, ascending=not asc)
        threshold_val = sorted_df.iloc[target_count-1][metric]
        
        # Apply mask
        if asc:
            mask = res_df[metric] >= threshold_val
        else:
            mask = res_df[metric] <= threshold_val
            
        shobu_df = res_df[mask]
        normal_df = res_df[~mask]
        
        shobu_count = len(shobu_df)
        shobu_base_bet = shobu_df['base_bet_amt'].sum()
        shobu_base_return = shobu_df['base_return_amt'].sum()
        shobu_roi = shobu_base_return / shobu_base_bet if shobu_base_bet > 0 else 0
        
        for mult in multipliers:
            total_bet = normal_df['base_bet_amt'].sum() + (shobu_base_bet * mult)
            total_return = normal_df['base_return_amt'].sum() + (shobu_base_return * mult)
            overall_roi = total_return / total_bet
            overall_profit = total_return - total_bet
            diff_profit = overall_profit - base_profit
            
            report.append(f"| {mult}倍 | {shobu_count} ({shobu_count/total_betted:.1%}) | {shobu_roi:.1%} | {overall_roi:.2%} | {overall_profit:,.0f}円 | {diff_profit:+,.0f}円 |")
            
            if overall_profit > best_profit:
                best_profit = overall_profit
                best_strategy = {
                    'metric': metric,
                    'threshold': threshold_val,
                    'mult': mult,
                    'shobu_roi': shobu_roi,
                    'overall_roi': overall_roi,
                    'profit': overall_profit,
                    'diff': diff_profit
                }
        report.append("\n")
        
    report.append(f"## 🏆 最適戦略\n")
    report.append(f"- **トリガー指標**: `{best_strategy['metric']}`\n")
    report.append(f"- **閾値**: `{best_strategy['threshold']:.4f}` 以上\n")
    report.append(f"- **最適倍率**: `{best_strategy['mult']}倍`\n")
    report.append(f"- **勝負対象の素のROI**: `{best_strategy['shobu_roi']:.1%}`\n")
    report.append(f"- **適用後の全体ROI**: `{best_strategy['overall_roi']:.2%}`\n")
    report.append(f"- **最終利益**: `{best_strategy['profit']:,.0f}円` (通常時比: `{best_strategy['diff']:+,.0f}円`)\n")

    with open('shobu_gake_report.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    print("Report generated.")

if __name__ == '__main__':
    generate_shobu_gake_report()
