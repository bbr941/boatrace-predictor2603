import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import pickle
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
import itertools

# --- Configuration ---
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
SAMPLE_SIZE = 40000 
INITIAL_BUDGET = 100000
EV_THRESHOLD = 1.2
KELLY_COEF = 0.5  # Half-Kelly

def load_data(limit=SAMPLE_SIZE):
    print(f"Loading latest {limit} rows from {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    # Corrected Query based on train_v3.py
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
    ORDER BY r.race_id DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Preprocessing
    df = df.sort_values(['race_id', 'boat_number']).reset_index(drop=True)
    df['target'] = (df['rank'] == 1).astype(int)
    df['rank_target'] = (6 - df['rank']).astype(int)
    
    # Feature Engineering
    base_num_cols = ['exhibition_time', 'exhibition_start_timing', 'pred_course', 'nat_win_rate', 'motor_rate', 'boat_rate']
    for col in base_num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    def zscore(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else 0
    df['exhibition_time_z'] = df.groupby('race_id')['exhibition_time'].transform(zscore)
    df['nat_win_rate_z'] = df.groupby('race_id')['nat_win_rate'].transform(zscore)
    
    # B1 diff
    b1_rates = df[df['boat_number'] == 1][['race_id', 'nat_win_rate']].rename(columns={'nat_win_rate': 'b1_rate'})
    df = df.merge(b1_rates, on='race_id', how='left')
    df['win_rate_diff_b1'] = df['nat_win_rate'] - df['b1_rate'].fillna(df['nat_win_rate'])
    df.drop('b1_rate', axis=1, inplace=True)
    
    return df

def train_pipeline(df_train, df_val, mode='advanced'):
    print(f"\n--- Training Pipeline: {mode.upper()} ---")
    if mode == 'advanced':
        features = ['venue_code', 'exhibition_time', 'exhibition_start_timing', 'pred_course', 
                    'nat_win_rate', 'motor_rate', 'boat_rate',
                    'exhibition_time_z', 'nat_win_rate_z', 'win_rate_diff_b1']
        global_mean = df_train['target'].mean()
        racer_stats = df_train.groupby('racer_id')['target'].agg(['mean', 'count'])
        m = 10
        racer_stats['smoothed'] = (racer_stats['count'] * racer_stats['mean'] + m * global_mean) / (racer_stats['count'] + m)
        te_map = racer_stats['smoothed'].to_dict()
        df_train['racer_target_enc'] = df_train['racer_id'].map(te_map).fillna(global_mean)
        df_val['racer_target_enc'] = df_val['racer_id'].map(te_map).fillna(global_mean)
        features.append('racer_target_enc')
    else:
        features = ['venue_code', 'exhibition_time', 'exhibition_start_timing', 'pred_course', 
                    'nat_win_rate', 'motor_rate', 'boat_rate']
    
    X_train = df_train[features].copy()
    y_train = df_train['target']
    y_train_rank = df_train['rank_target']
    X_val = df_val[features].copy()
    y_val = df_val['target']
    y_val_rank = df_val['rank_target']
    
    cat_features = ['venue_code']
    for f in cat_features:
        X_train[f] = X_train[f].astype('category')
        X_val[f] = X_val[f].astype('category')
        
    lgb_params = {
        'objective': 'lambdarank' if mode == 'advanced' else 'binary',
        'metric': 'ndcg' if mode == 'advanced' else 'binary_logloss',
        'verbose': -1,
        'learning_rate': 0.05,
        'num_leaves': 31,
    }
    
    train_ds = lgb.Dataset(X_train, label=y_train_rank if mode == 'advanced' else y_train, 
                           categorical_feature=cat_features, free_raw_data=False)
    val_ds = lgb.Dataset(X_val, label=y_val_rank if mode == 'advanced' else y_val, 
                         reference=train_ds, categorical_feature=cat_features, free_raw_data=False)
    
    if mode == 'advanced':
        train_groups = df_train.groupby('race_id', sort=False)['race_id'].count().values
        val_groups = df_val.groupby('race_id', sort=False)['race_id'].count().values
        train_ds.set_group(train_groups)
        val_ds.set_group(val_groups)
        
    model_lgb = lgb.train(lgb_params, train_ds, valid_sets=[val_ds], 
                          num_boost_round=150, callbacks=[lgb.early_stopping(30)])
    
    model_cb = cb.CatBoostClassifier(iterations=150, depth=6, learning_rate=0.05, verbose=0)
    model_cb.fit(X_train, y_train, cat_features=cat_features)
    
    if mode == 'advanced':
        lgb_scores = model_lgb.predict(X_val)
        def softmax_cal(scores, groups):
            probs = np.zeros_like(scores)
            idx = 0
            for size in groups:
                s = scores[idx:idx+size]
                e_x = np.exp(s - np.max(s))
                probs[idx:idx+size] = e_x / e_x.sum()
                idx += size
            return probs
        lgb_preds = softmax_cal(lgb_scores, val_groups)
    else:
        lgb_preds = model_lgb.predict(X_val)
    
    cb_preds = model_cb.predict_proba(X_val)[:, 1]
    
    X_meta = np.vstack([lgb_preds, cb_preds]).T
    meta_model = LogisticRegression().fit(X_meta, y_val)
    print(f"  {mode.upper()} trained.")
    
    return mode, features, model_lgb, model_cb, meta_model

def predict_meta(X_test, groups, pipeline_tuple):
    mode, features, model_lgb, model_cb, meta_model = pipeline_tuple
    X = X_test[features].copy()
    X['venue_code'] = X['venue_code'].astype('category')
    
    if mode == 'advanced':
        lgb_scores = model_lgb.predict(X)
        probs = np.zeros_like(lgb_scores)
        idx = 0
        for size in groups:
            s = lgb_scores[idx:idx+size]
            e_x = np.exp(s - np.max(s))
            probs[idx:idx+size] = e_x / (e_x.sum() + 1e-9)
            idx += size
        lgb_preds = probs
    else:
        lgb_preds = model_lgb.predict(X)
    cb_preds = model_cb.predict_proba(X)[:, 1]
    X_meta = np.vstack([lgb_preds, cb_preds]).T
    return meta_model.predict_proba(X_meta)[:, 1]

def calculate_trifecta_probs(win_probs):
    combos = list(itertools.permutations(range(1, 7), 3))
    t_probs = {}
    for c in combos:
        p1 = win_probs[c[0]-1]
        p2 = win_probs[c[1]-1] / (1 - p1 + 1e-9)
        p3 = win_probs[c[2]-1] / (1 - p1 - win_probs[c[1]-1] + 1e-9)
        # Result "123" to match odds_data.combination
        t_probs[f"{c[0]}{c[1]}{c[2]}"] = max(0, p1 * p2 * p3)
    total = sum(t_probs.values())
    return {k: v / (total + 1e-9) for k, v in t_probs.items()}

def run_simulation(df_test, pipeline_tuple, odds_df, payoff_df, name="Model"):
    print(f"\n--- Running ROI Simulation for {name} ---")
    balance = INITIAL_BUDGET
    history = [balance]
    races = df_test['race_id'].unique()
    total_invested = 0
    total_returned = 0
    bets_count = 0
    hits_count = 0
    
    for rid in races:
        race_rows = df_test[df_test['race_id'] == rid]
        if len(race_rows) != 6: continue
        win_probs = predict_meta(race_rows, [6], pipeline_tuple)
        trifecta_probs = calculate_trifecta_probs(win_probs)
        race_odds = odds_df[odds_df['race_id'] == rid]
        if race_odds.empty: continue
        race_payoff = payoff_df[payoff_df['race_id'] == rid]
        if race_payoff.empty: continue
        winner_combo_hypen = race_payoff['sanrentan_result'].values[0] # "1-2-3"
        # Convert "1-2-3" to "123" for matching
        winner_combo = winner_combo_hypen.replace('-', '')
        actual_payoff = race_payoff['sanrentan_payoff'].values[0]
        
        for combo, prob in trifecta_probs.items():
            o_row = race_odds[race_odds['combination'] == combo]
            if o_row.empty: continue
            odds = o_row['odds_5min'].values[0]
            if odds <= 1: continue
            ev = prob * odds
            if ev > EV_THRESHOLD:
                b = odds - 1
                f = (b * prob - (1 - prob)) / b
                ratio = max(0, f * KELLY_COEF)
                if ratio > 0:
                    amt = int(balance * ratio // 100 * 100)
                    if amt < 100: continue
                    if balance < amt: amt = balance
                    balance -= amt
                    total_invested += amt
                    bets_count += 1
                    if combo == winner_combo:
                        ret = amt * (actual_payoff / 100)
                        balance += ret
                        total_returned += ret
                        hits_count += 1
        history.append(balance)
        if balance <= 0: break
    
    roi = (total_returned / total_invested * 100) if total_invested > 0 else 0
    max_dd = 0
    peak = history[0]
    for p in history:
        if p > peak: peak = p
        dd = (peak - p) / peak if peak > 0 else 0
        if dd > max_dd: max_dd = dd
    return history, roi, balance - INITIAL_BUDGET, max_dd

def main():
    df = load_data(40000)
    split_idx = int(len(df) * 0.7)
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    conn = sqlite3.connect(DB_PATH)
    rids = df_test['race_id'].unique()
    rids_str = "('" + "','".join(df_test['race_id'].unique()) + "')"
    odds_df = pd.read_sql(f"SELECT * FROM odds_data WHERE race_id IN {rids_str}", conn)
    payoff_df = pd.read_sql(f"SELECT race_id, sanrentan_result, sanrentan_payoff FROM payoffs WHERE race_id IN {rids_str}", conn)
    conn.close()
    
    pipe_legacy = train_pipeline(df_train.copy(), df_test.copy(), mode='legacy')
    pipe_adv = train_pipeline(df_train.copy(), df_test.copy(), mode='advanced')
    
    # We need to re-apply TE to df_test for the simulation loop if it wasn't there
    # It was applied in train_pipeline for df_val (which is df_test here)
    # But wait, df_test needs racer_target_enc for the simulation loop
    # Let's check if pipe_adv modified df_test
    # Actually, let's just redo TE on df_test here to be safe
    global_mean = df_train['target'].mean()
    te_map = df_train.groupby('racer_id')['target'].mean().to_dict()
    df_test['racer_target_enc'] = df_test['racer_id'].map(te_map).fillna(global_mean)
    
    h_leg, r_leg, p_leg, d_leg = run_simulation(df_test, pipe_legacy, odds_df, payoff_df, "V3.1 Legacy")
    h_adv, r_adv, p_adv, d_adv = run_simulation(df_test, pipe_adv, odds_df, payoff_df, "V3.1+ Advanced")
    
    plt.figure(figsize=(12, 6))
    plt.plot(h_leg, label=f'Legacy (ROI: {r_leg:.1f}%, DD: {d_leg:.1%})', color='gray')
    plt.plot(h_adv, label=f'Advanced (ROI: {r_adv:.1f}%, DD: {d_adv:.1%})', color='red', linewidth=2)
    plt.axhline(INITIAL_BUDGET, color='black', linestyle='--')
    plt.title("ROI Comparison: V3.1 Legacy vs Advanced (Out-of-Sample)")
    plt.xlabel("Races")
    plt.ylabel("Cumulative Balance (JPY)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('roi_comparison.png')
    
    print("\n" + "="*40)
    print("      FINAL ROI COMPARISON REPORT")
    print("="*40)
    print(f"{'Metric':<20} | {'Legacy':<10} | {'Advanced':<10}")
    print("-"*40)
    print(f"{'Total Profit':<20} | ¥{p_leg:,.0f} | ¥{p_adv:,.0f}")
    print(f"{'ROI':<20} | {r_leg:>9.2f}% | {r_adv:>9.2f}%")
    print(f"{'Max Drawdown':<20} | {d_leg:>9.2%} | {d_adv:>9.2%}")
    print("-" * 40)

if __name__ == "__main__":
    main()
