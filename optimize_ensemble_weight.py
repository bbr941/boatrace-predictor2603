"""
optimize_ensemble_weight.py
ボートレース アンサンブル最適化: Logit (対数オッズ) 空間結合によるブレンドウェイト探索
"""

import os
import sys
import time
import math
import json
import sqlite3
import argparse
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

import train_model
from odds_normalizer import probs_to_init_scores
from probability_calibration import (
    get_default_calibrator,
    load_probability_config
)
from portfolio_optimizer import PortfolioOptimizer, load_correlation_mask

# Optuna のログ出力を警告以上のみに抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# デフォルト設定 & パス
# ==========================================
MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DB_PATH = 'boatrace.db' if os.path.exists('boatrace.db') else r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
ENSEMBLE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'app_data', 'ensemble_config.json')

# 最適化設定
prob_config = load_probability_config()
D2_PARAM = prob_config.get('d2', 0.40)
D3_PARAM = prob_config.get('d3', 0.60)
P1_TH_DEFAULT = prob_config.get('p1_th', 0.49)
GAP_TH_DEFAULT = prob_config.get('gap_th', 0.010)

BANKROLL_DEFAULT = 100000.0
RISK_AVERSION_DEFAULT = 1.0
MAX_EXPOSURE_DEFAULT = 0.05
MAX_CONCENTRATION_DEFAULT = 0.02
MIN_EV_DEFAULT = 1.0


def get_db_connection():
    if os.path.exists('boatrace.db'):
        return sqlite3.connect('boatrace.db')
    return sqlite3.connect(DB_PATH)


def load_all_odds_batch(conn, race_ids):
    """5,000レース分の3連単オッズデータを一括でメモリキャッシュ"""
    print("  [Pre-caching] オッズデータを一括ロード中...", flush=True)
    t0 = time.time()
    odds_cache = {}
    cursor = conn.cursor()
    chunk_size = 500
    for i in range(0, len(race_ids), chunk_size):
        chunk_rids = race_ids[i:i + chunk_size]
        placeholders = ','.join(['?'] * len(chunk_rids))
        query = f"SELECT race_id, combination, odds_1min FROM odds_data WHERE race_id IN ({placeholders}) AND length(combination) = 3"
        cursor.execute(query, chunk_rids)
        for rid, comb_db, val in cursor.fetchall():
            comb_str = str(comb_db)
            if len(comb_str) == 3:
                fmt_comb = f"{comb_str[0]}-{comb_str[1]}-{comb_str[2]}"
                if rid not in odds_cache:
                    odds_cache[rid] = {}
                odds_cache[rid][fmt_comb] = float(val) if val is not None else 0.0
    print(f"        -> {len(odds_cache):,} レース分のオッズデータをキャッシュ完了 ({time.time() - t0:.2f}秒)", flush=True)
    return odds_cache


def fast_benter_probs(p_norm, boats, d2=0.40, d3=0.60):
    """
    正規化された1着確率 p_norm (6艇) から Benter モデルによる 120通り3連単確率を展開する。
    戻り値: benter_probs (list of dict), max_p1 (float), prob_gap (float)
    """
    n_boats = len(boats)
    p1_dict = {boats[i]: float(p_norm[i]) for i in range(n_boats)}
    
    # Benter べき乗
    p1_d2 = {boats[i]: max(float(p_norm[i]), 1e-9) ** d2 for i in range(n_boats)}
    p1_d3 = {boats[i]: max(float(p_norm[i]), 1e-9) ** d3 for i in range(n_boats)}
    
    # 120通り展開
    combos = list(itertools.permutations(boats, 3))
    benter_probs = []
    
    for c in combos:
        b1, b2, b3 = c
        prob1 = p1_dict[b1]
        
        denom2 = sum(p1_d2[b] for b in boats if b != b1)
        prob2 = p1_d2[b2] / denom2 if denom2 > 0 else 1e-9
        
        denom3 = sum(p1_d3[b] for b in boats if b != b1 and b != b2)
        prob3 = p1_d3[b3] / denom3 if denom3 > 0 else 1e-9
        
        total_prob = prob1 * prob2 * prob3
        benter_probs.append({'combo': f"{b1}-{b2}-{b3}", 'prob': float(total_prob)})
        
    benter_probs.sort(key=lambda x: x['prob'], reverse=True)
    
    p1_sorted = np.sort(p_norm)[::-1]
    max_p1 = float(p1_sorted[0])
    prob_gap = float(p1_sorted[0] - p1_sorted[1]) if len(p1_sorted) >= 2 else 0.0
    
    return benter_probs, max_p1, prob_gap


def evaluate_ensemble(
    w: float,
    cached_races: list,
    odds_cache: dict,
    optimizer: PortfolioOptimizer,
    d2: float = 0.40,
    d3: float = 0.60,
    p1_th: float = 0.49,
    gap_th: float = 0.010,
    bankroll: float = 100000.0,
    risk_aversion: float = 1.0,
    max_exposure: float = 0.05,
    max_concentration: float = 0.02,
    min_ev: float = 1.0
) -> dict:
    """
    指定されたブレンドウェイト w に対するバックテストを実行し、評価指標を算出する。
    Z_hybrid = (1 - w) * Z_Lambda + w * Z_Residual
    P_raw = 1 / (1 + exp(-Z_hybrid))
    P_norm = P_raw / sum(P_raw)
    """
    stats = {'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0}
    betted_profits = []
    profit_series = []

    for race in cached_races:
        rid = race['race_id']
        boats = race['boats']
        z_l = race['z_lambda']
        z_r = race['z_residual']
        actual_combo = race['actual_combo']
        
        # 1. Logit 空間結合
        z_hyb = (1.0 - w) * z_l + w * z_r
        
        # 2. シグモイド変換 & レース内 6艇合計=1.0 正規化
        p_raw = 1.0 / (1.0 + np.exp(-np.clip(z_hyb, -30, 30)))
        sum_p = np.sum(p_raw)
        if sum_p <= 0:
            p_norm = np.ones(len(boats)) / len(boats)
        else:
            p_norm = p_raw / sum_p
            
        # 3. 高速 Plan B フィルター判定
        p_sorted = np.sort(p_norm)[::-1]
        max_p1 = p_sorted[0]
        prob_gap = p_sorted[0] - p_sorted[1] if len(p_sorted) >= 2 else 0.0
        
        if not (max_p1 >= p1_th and prob_gap >= gap_th):
            profit_series.append(0.0)
            continue
            
        # 4. オッズ取得
        all_odds = odds_cache.get(rid)
        if not all_odds:
            profit_series.append(0.0)
            continue
            
        # 5. Benter 確率展開
        benter_probs, _, _ = fast_benter_probs(p_norm, boats, d2=d2, d3=d3)
        
        # 6. ポートフォリオ最適化
        probs_dict = {p['combo']: p['prob'] for p in benter_probs}
        bets = optimizer.optimize_funds(
            probabilities=probs_dict,
            odds=all_odds,
            bankroll=bankroll,
            risk_aversion=risk_aversion,
            max_exposure=max_exposure,
            max_concentration=max_concentration,
            min_ev=min_ev
        )
        
        if not bets:
            profit_series.append(0.0)
            continue
            
        bet_sum = sum(bets.values())
        ret_sum = 0.0
        hit = False
        if actual_combo and actual_combo in bets:
            hit = True
            ret_sum = bets[actual_combo] * all_odds.get(actual_combo, 0.0)
            
        stats['betted'] += 1
        stats['bet_amt'] += bet_sum
        stats['return_amt'] += ret_sum
        if hit:
            stats['hits'] += 1
            
        pnl = ret_sum - bet_sum
        profit_series.append(pnl)
        betted_profits.append(pnl)

    # 評価指標計算
    total_bet = stats['bet_amt']
    total_return = stats['return_amt']
    total_profit = total_return - total_bet
    roi = (total_return / total_bet) if total_bet > 0 else 0.0
    hit_rate = (stats['hits'] / stats['betted']) if stats['betted'] > 0 else 0.0

    # ドローダウン
    cum_pnl = np.cumsum(profit_series)
    peak_pnl = np.maximum.accumulate(cum_pnl)
    drawdown_amt = peak_pnl - cum_pnl
    mdd_amt = float(np.max(drawdown_amt)) if len(drawdown_amt) > 0 else 0.0
    
    equity_series = bankroll + cum_pnl
    peak_equity = np.maximum.accumulate(equity_series)
    pct_drawdown = (peak_equity - equity_series) / np.maximum(peak_equity, 1e-9)
    mdd_pct = float(np.max(pct_drawdown) * 100.0) if len(pct_drawdown) > 0 else 0.0

    # シャープレシオ
    if len(betted_profits) > 1:
        mean_p = float(np.mean(betted_profits))
        std_p = float(np.std(betted_profits, ddof=1))
        sharpe_per_race = (mean_p / std_p) if std_p > 0 else 0.0
        annualized_sharpe = sharpe_per_race * np.sqrt(len(betted_profits))
    else:
        sharpe_per_race = 0.0
        annualized_sharpe = 0.0

    return {
        'w': w,
        'betted': stats['betted'],
        'hits': stats['hits'],
        'hit_rate': hit_rate,
        'total_bet': total_bet,
        'total_return': total_return,
        'total_profit': total_profit,
        'roi': roi,
        'mdd_amt': mdd_amt,
        'mdd_pct': mdd_pct,
        'sharpe': sharpe_per_race,
        'annualized_sharpe': annualized_sharpe
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize Ensemble Blend Weight in Logit Space")
    parser.add_argument('--races', type=int, default=5000, help="Number of races (default: 5000)")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument('--bankroll', type=float, default=BANKROLL_DEFAULT, help="Bankroll in JPY")
    parser.add_argument('--risk_aversion', type=float, default=RISK_AVERSION_DEFAULT, help="Risk aversion lambda")
    parser.add_argument('--max_exposure', type=float, default=MAX_EXPOSURE_DEFAULT, help="Max exposure per race")
    parser.add_argument('--max_concentration', type=float, default=MAX_CONCENTRATION_DEFAULT, help="Max concentration per combo")
    parser.add_argument('--min_ev', type=float, default=MIN_EV_DEFAULT, help="Minimum EV threshold")
    parser.add_argument('--p1_th', type=float, default=P1_TH_DEFAULT, help="P1 confidence threshold")
    parser.add_argument('--gap_th', type=float, default=GAP_TH_DEFAULT, help="Prob gap threshold")
    parser.add_argument('--d2', type=float, default=D2_PARAM, help="Benter d2 parameter")
    parser.add_argument('--d3', type=float, default=D3_PARAM, help="Benter d3 parameter")

    args = parser.parse_args()
    t_start = time.time()

    print("\n" + "=" * 75, flush=True)
    print("  🚀 BOATRACE アンサンブル最適化: Logit空間結合探索 (Optuna Trials=50)", flush=True)
    print("=" * 75, flush=True)
    print(f"  本命モデル (LambdaRank): {MODEL_HONMEI_PATH} (Platt Scaling -> Logit)", flush=True)
    print(f"  残差モデル (Residual)  : {MODEL_RESIDUAL_PATH} (init_score + Δz)", flush=True)
    print(f"  対象レース数           : {args.races:,} レース", flush=True)
    print(f"  Benter 設定            : d2 = {args.d2:.2f}, d3 = {args.d3:.2f}", flush=True)
    print(f"  Plan B フィルター      : P1 >= {args.p1_th:.2f}, ΔP >= {args.gap_th:.3f}", flush=True)
    print(f"  ポートフォリオ設定     : λ = {args.risk_aversion:.1f}, Max Exposure = {args.max_exposure:.1%}", flush=True)
    print("-" * 75, flush=True)

    # 1. DBからレースID取得 & オッズ一括ロード
    print("  [1/4] DBから対象レース & オッズデータを一括キャッシュ中...", flush=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT ?", [args.races])
    valid_races = [row[0] for row in cursor.fetchall()]
    valid_races_set = set(valid_races)
    print(f"        -> {len(valid_races):,} レース特定完了", flush=True)

    odds_cache = load_all_odds_batch(conn, valid_races)
    conn.close()

    # 2. CSVからデータ抽出 & 前処理
    print("  [2/4] CSVからレース特徴量を高速抽出 & 前処理...", flush=True)
    t0_csv = time.time()
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=150000):
        matched = chunk[chunk['race_id'].isin(valid_races_set)]
        if len(matched) > 0:
            chunks.append(matched)
            
    df_matched = pd.concat(chunks, ignore_index=True)
    test_df = train_model.preprocess_data(df_matched)
    print(f"        -> {len(test_df):,} 行 前処理完了 ({time.time() - t0_csv:.2f}秒)", flush=True)

    # オッズベースマージン算出
    if 'syn_win_rate' not in test_df.columns:
        test_df['syn_win_rate'] = 0.0
    race_sums = test_df.groupby('race_id')['syn_win_rate'].transform('sum')
    has_valid_odds = (race_sums > 0) & np.isfinite(race_sums)
    p_norm = np.where(has_valid_odds, test_df['syn_win_rate'] / np.maximum(race_sums, 1e-9), 1.0 / 6.0)
    test_df['init_score'] = probs_to_init_scores(p_norm, clip_eps=1e-5)

    # 3. 両モデルの推論 & Logit 変換
    print("  [3/4] 両モデルの推論 & Logit 空間への変換中...", flush=True)
    
    # 3.1 本命モデル (Platt Scaling -> Logit)
    model_honmei = lgb.Booster(model_file=MODEL_HONMEI_PATH)
    feats_honmei = model_honmei.feature_name()
    for f in feats_honmei:
        if f not in test_df.columns: test_df[f] = 0
    test_df['score_honmei'] = model_honmei.predict(test_df[feats_honmei])

    calibrator = get_default_calibrator('platt')
    z_lambda_map = {}
    for rid, grp in test_df.groupby('race_id', sort=False):
        s_dict = dict(zip(grp['boat_number'], grp['score_honmei']))
        p_dict = calibrator.calibrate_scores(s_dict)
        for b in grp['boat_number']:
            p_val = np.clip(p_dict[b], 1e-5, 1.0 - 1e-5)
            z_lambda_map[(rid, b)] = float(np.log(p_val / (1.0 - p_val)))

    test_df['z_lambda'] = [z_lambda_map[(r, b)] for r, b in zip(test_df['race_id'], test_df['boat_number'])]

    # 3.2 残差モデル (init_score + Δz)
    model_residual = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
    feats_residual = model_residual.feature_name()
    for f in feats_residual:
        if f not in test_df.columns: test_df[f] = 0
    raw_res = model_residual.predict(test_df[feats_residual], raw_score=True)
    test_df['z_residual'] = raw_res + test_df['init_score'].to_numpy()

    # メモリ上にレースオブジェクトをキャッシュ
    cached_races = []
    for rid, group in test_df.groupby('race_id', sort=False):
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            actual_combo = f"{int(r1)}-{int(r2)}-{int(r3)}"
        except IndexError:
            actual_combo = None

        cached_races.append({
            'race_id': rid,
            'boats': group['boat_number'].to_numpy(),
            'z_lambda': group['z_lambda'].to_numpy(dtype=float),
            'z_residual': group['z_residual'].to_numpy(dtype=float),
            'actual_combo': actual_combo
        })

    print(f"        -> {len(cached_races):,} レースをメモリキャッシュ完了", flush=True)

    # 4. Optuna による探索
    print("\n  [4/4] Optuna による最適なブレンド比率 w の探索開始 (Logit空間)...", flush=True)
    optimizer = PortfolioOptimizer()

    def objective(trial):
        w = trial.suggest_float('w', 0.0, 1.0, step=0.02)
        res = evaluate_ensemble(
            w=w,
            cached_races=cached_races,
            odds_cache=odds_cache,
            optimizer=optimizer,
            d2=args.d2,
            d3=args.d3,
            p1_th=args.p1_th,
            gap_th=args.gap_th,
            bankroll=args.bankroll,
            risk_aversion=args.risk_aversion,
            max_exposure=args.max_exposure,
            max_concentration=args.max_concentration,
            min_ev=args.min_ev
        )
        if res['betted'] < 50:
            return -1e7
        trial.set_user_attr('roi', res['roi'])
        trial.set_user_attr('betted', res['betted'])
        trial.set_user_attr('hit_rate', res['hit_rate'])
        trial.set_user_attr('sharpe', res['sharpe'])
        return res['total_profit']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    # 代表グリッドを初期投入
    for w_init in [0.0, 1.0, 0.5, 0.2, 0.8, 0.3, 0.7]:
        study.enqueue_trial({'w': w_init})

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best_w = study.best_params['w']
    best_profit = study.best_value

    # 代表ウェイトの詳細評価
    print("\n" + "=" * 75, flush=True)
    print("  📊 アンサンブル最適化 検証結果サマリー (Logit空間結合)", flush=True)
    print("=" * 75, flush=True)

    eval_w0 = evaluate_ensemble(0.0, cached_races, odds_cache, optimizer, d2=args.d2, d3=args.d3, p1_th=args.p1_th, gap_th=args.gap_th, bankroll=args.bankroll, risk_aversion=args.risk_aversion, max_exposure=args.max_exposure, max_concentration=args.max_concentration, min_ev=args.min_ev)
    eval_w1 = evaluate_ensemble(1.0, cached_races, odds_cache, optimizer, d2=args.d2, d3=args.d3, p1_th=args.p1_th, gap_th=args.gap_th, bankroll=args.bankroll, risk_aversion=args.risk_aversion, max_exposure=args.max_exposure, max_concentration=args.max_concentration, min_ev=args.min_ev)
    eval_best = evaluate_ensemble(best_w, cached_races, odds_cache, optimizer, d2=args.d2, d3=args.d3, p1_th=args.p1_th, gap_th=args.gap_th, bankroll=args.bankroll, risk_aversion=args.risk_aversion, max_exposure=args.max_exposure, max_concentration=args.max_concentration, min_ev=args.min_ev)

    print(f"  ┌───────────────────────┬────────────────────┬────────────────────┬────────────────────┐", flush=True)
    print(f"  │ 項目                  │ w=0.0 (Pure Honmei)│ w=1.0 (Pure Resid) │ w={best_w:.2f} (最適化提案)  │", flush=True)
    print(f"  ├───────────────────────┼────────────────────┼────────────────────┼────────────────────┤", flush=True)
    print(f"  │ 参戦レース数          │  {eval_w0['betted']:>5d} レース     │  {eval_w1['betted']:>5d} レース     │  {eval_best['betted']:>5d} レース     │", flush=True)
    print(f"  │ 的中レース数 (的中率) │  {eval_w0['hits']:>3d} ({eval_w0['hit_rate']:>5.2%})     │  {eval_w1['hits']:>3d} ({eval_w1['hit_rate']:>5.2%})     │  {eval_best['hits']:>3d} ({eval_best['hit_rate']:>5.2%})     │", flush=True)
    print(f"  │ 総投資金額            │  {int(eval_w0['total_bet']):>10,d} 円   │  {int(eval_w1['total_bet']):>10,d} 円   │  {int(eval_best['total_bet']):>10,d} 円   │", flush=True)
    print(f"  │ 総払戻金額            │  {int(eval_w0['total_return']):>10,d} 円   │  {int(eval_w1['total_return']):>10,d} 円   │  {int(eval_best['total_return']):>10,d} 円   │", flush=True)
    print(f"  │ 最終損益 (Profit)     │ {int(eval_w0['total_profit']):>+11,d} 円   │ {int(eval_w1['total_profit']):>+11,d} 円   │ {int(eval_best['total_profit']):>+11,d} 円   │", flush=True)
    print(f"  │ 回収率 (ROI)          │        {eval_w0['roi']:>6.2%}      │        {eval_w1['roi']:>6.2%}      │        {eval_best['roi']:>6.2%}      │", flush=True)
    print(f"  │ 最大ドローダウン(MDD) │  {int(eval_w0['mdd_amt']):>10,d} 円   │  {int(eval_w1['mdd_amt']):>10,d} 円   │  {int(eval_best['mdd_amt']):>10,d} 円   │", flush=True)
    print(f"  │ シャープレシオ        │       {eval_w0['sharpe']:>7.4f}      │       {eval_w1['sharpe']:>7.4f}      │       {eval_best['sharpe']:>7.4f}      │", flush=True)
    print(f"  └───────────────────────┴────────────────────┴────────────────────┴────────────────────┘", flush=True)

    print(f"\n  🏆 最適ブレンド比率: w_residual = {best_w:.2f} (w_honmei = {1.0 - best_w:.2f})", flush=True)
    print(f"     Z_hybrid = {1.0 - best_w:.2f} * Z_Lambda + {best_w:.2f} * Z_Residual", flush=True)
    print(f"     総所要時間: {time.time() - t_start:.2f} 秒\n", flush=True)

    # 設定保存
    os.makedirs(os.path.dirname(ENSEMBLE_CONFIG_PATH), exist_ok=True)
    with open(ENSEMBLE_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'blend_space': 'logit',
            'best_w_residual': best_w,
            'best_w_honmei': 1.0 - best_w,
            'roi': eval_best['roi'],
            'profit': eval_best['total_profit'],
            'hit_rate': eval_best['hit_rate'],
            'betted_races': eval_best['betted'],
            'mdd_amt': eval_best['mdd_amt'],
            'sharpe': eval_best['sharpe']
        }, f, indent=4, ensure_ascii=False)
    print(f"  💾 最適アンサンブル設定を保存しました: {ENSEMBLE_CONFIG_PATH}", flush=True)
    print("=" * 75 + "\n", flush=True)


if __name__ == "__main__":
    main()
