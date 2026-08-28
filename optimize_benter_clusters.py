"""
optimize_benter_clusters.py
ボートレース: 会場クラスタ別 Benter パラメーター (d2, d3) 最適化スクリプト
- クラスタ0: [18, 21, 23, 24] (徳山, 芦屋, 唐津, 大村 などイン超強水面)
- クラスタ1: [2, 3, 4, 14, 22] (戸田, 江戸川, 平和島, 鳴門, 福岡 など難水面・イン受難)
- クラスタ2: その他の全会場 (標準水面)
- Gatekeeper (85th percentile / 上位15%) & Optimizer (EV >= 1.25, Odds <= 30.0, λ=1.0) 固定
- 物理制約: 全クラスタで d3 >= d2
- Optuna Trials=100 による 6変数同時最適化
"""

import os
import sys
import time
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
from probability_calibration import get_default_calibrator
from portfolio_optimizer import PortfolioOptimizer, load_correlation_mask

# Optuna ログ抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# デフォルト設定 & パス
# ==========================================
MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DB_PATH = 'boatrace.db' if os.path.exists('boatrace.db') else r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
OUTPUT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'app_data', 'benter_cluster_config.json')

# 会場クラスタ定義
CLUSTER_0_VENUES = {18, 21, 23, 24}
CLUSTER_1_VENUES = {2, 3, 4, 14, 22}

def get_venue_cluster(venue_code: int) -> int:
    """会場コードからクラスタID (0, 1, 2) を判定"""
    if venue_code in CLUSTER_0_VENUES:
        return 0
    elif venue_code in CLUSTER_1_VENUES:
        return 1
    else:
        return 2


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
    正規化された1着確率 p_norm (6艇) から指定の d2, d3 で120通り3連単確率を展開
    """
    n_boats = len(boats)
    p1_dict = {boats[i]: float(p_norm[i]) for i in range(n_boats)}
    
    p1_d2 = {boats[i]: max(float(p_norm[i]), 1e-9) ** d2 for i in range(n_boats)}
    p1_d3 = {boats[i]: max(float(p_norm[i]), 1e-9) ** d3 for i in range(n_boats)}
    
    combos = list(itertools.permutations(boats, 3))
    probs_dict = {}
    
    for c in combos:
        b1, b2, b3 = c
        prob1 = p1_dict[b1]
        
        denom2 = sum(p1_d2[b] for b in boats if b != b1)
        prob2 = p1_d2[b2] / denom2 if denom2 > 0 else 1e-9
        
        denom3 = sum(p1_d3[b] for b in boats if b != b1 and b != b2)
        prob3 = p1_d3[b3] / denom3 if denom3 > 0 else 1e-9
        
        probs_dict[f"{b1}-{b2}-{b3}"] = float(prob1 * prob2 * prob3)
        
    return probs_dict


def evaluate_cluster_params(
    cluster_params: dict,
    gatekeeper_passed_races: list,
    odds_cache: dict,
    optimizer: PortfolioOptimizer,
    bankroll: float = 100000.0,
    risk_aversion: float = 1.0,
    max_exposure: float = 0.05,
    max_concentration: float = 0.02,
    min_ev: float = 1.25,
    max_odds: float = 30.0
) -> dict:
    """
    各クラスタの (d2, d3) を適用してバックテストを実行し、ROIや利益を算出
    """
    stats = {'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0}
    cluster_stats = {
        0: {'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0},
        1: {'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0},
        2: {'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0}
    }
    betted_profits = []
    profit_series = []

    for race in gatekeeper_passed_races:
        rid = race['race_id']
        c_id = race['cluster_id']
        boats = race['boats']
        p_norm = race['p_norm_residual']
        actual_combo = race['actual_combo']
        
        d2, d3 = cluster_params[c_id]
        
        # Benter 展開
        probs_dict = fast_benter_probs(p_norm, boats, d2=d2, d3=d3)
        
        all_odds = odds_cache.get(rid, {})
        if not all_odds:
            profit_series.append(0.0)
            continue
            
        # ポートフォリオ最適化
        bets = optimizer.optimize_funds(
            probabilities=probs_dict,
            odds=all_odds,
            bankroll=bankroll,
            risk_aversion=risk_aversion,
            max_exposure=max_exposure,
            max_concentration=max_concentration,
            min_ev=min_ev,
            max_odds=max_odds
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
            
        cluster_stats[c_id]['betted'] += 1
        cluster_stats[c_id]['bet_amt'] += bet_sum
        cluster_stats[c_id]['return_amt'] += ret_sum
        if hit:
            cluster_stats[c_id]['hits'] += 1
            
        pnl = ret_sum - bet_sum
        profit_series.append(pnl)
        betted_profits.append(pnl)

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

    return {
        'roi': roi,
        'total_profit': total_profit,
        'total_bet': total_bet,
        'total_return': total_return,
        'betted': stats['betted'],
        'hits': stats['hits'],
        'hit_rate': hit_rate,
        'mdd_amt': mdd_amt,
        'cluster_stats': cluster_stats
    }


def main():
    parser = argparse.ArgumentParser(description="Optimize Benter parameters per venue cluster")
    parser.add_argument('--races', type=int, default=5000, help="Number of races to simulate (default: 5000)")
    parser.add_argument('--n_trials', type=int, default=100, help="Number of Optuna trials (default: 100)")
    parser.add_argument('--bankroll', type=float, default=100000.0, help="Bankroll in JPY")
    parser.add_argument('--risk_aversion', type=float, default=1.0, help="Risk aversion lambda")
    parser.add_argument('--max_exposure', type=float, default=0.05, help="Max exposure per race")
    parser.add_argument('--max_concentration', type=float, default=0.02, help="Max concentration per combo")
    parser.add_argument('--min_ev', type=float, default=1.25, help="Minimum EV threshold")
    parser.add_argument('--max_odds', type=float, default=30.0, help="Maximum Odds upper bound")
    parser.add_argument('--percentile', type=float, default=85.0, help="Gatekeeper percentile threshold")

    args = parser.parse_args()
    t_start = time.time()

    print("\n" + "=" * 75, flush=True)
    print("  🚀 BOATRACE 会場クラスタ別 Benter パラメーター最適化 (Optuna Trials=100)", flush=True)
    print("=" * 75, flush=True)
    print(f"  対象レース数           : {args.races:,} レース", flush=True)
    print(f"  Gatekeeper スクリーニング: 上位 15.0% 分位点 (85th percentile 動的閾値)", flush=True)
    print(f"  Optimizer 厳格化制約   : EV >= {args.min_ev:.2f}, Odds <= {args.max_odds:.1f}, λ = {args.risk_aversion:.1f}", flush=True)
    print(f"  会場クラスタ定義       :")
    print(f"    ・Cluster 0 (イン超強) : {sorted(list(CLUSTER_0_VENUES))} (徳山, 芦屋, 唐津, 大村)")
    print(f"    ・Cluster 1 (難水面)   : {sorted(list(CLUSTER_1_VENUES))} (戸田, 江戸川, 平和島, 鳴門, 福岡)")
    print(f"    ・Cluster 2 (標準水面) : その他の全会場 (15場)")
    print(f"  物理制約               : 全クラスタで d3 >= d2 (d3 < d2 は強ペナルティ排除)", flush=True)
    print("-" * 75, flush=True)

    # 1. DBからレースID取得 (rowid DESCにより全会場の直近レースを均等取得) & オッズ一括ロード
    print("  [1/4] DBから対象レース & 最新オッズデータを一括キャッシュ中...", flush=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY rowid DESC LIMIT ?", [args.races])
    valid_races = [row[0] for row in cursor.fetchall()]
    valid_races_set = set(valid_races)
    print(f"        -> 抽出レース数: {len(valid_races):,} レース", flush=True)

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
    
    # 会場コードマッピングの保存 (前処理前に取得)
    race_venue_map = {}
    for rid, vcode in zip(df_matched['race_id'], df_matched['venue_code']):
        if rid not in race_venue_map:
            race_venue_map[rid] = int(vcode)

    test_df = train_model.preprocess_data(df_matched)
    print(f"        -> 抽出行数: {len(test_df):,} 行 ({time.time() - t0_csv:.2f}秒)", flush=True)

    # オッズベースマージン算出
    if 'syn_win_rate' not in test_df.columns:
        test_df['syn_win_rate'] = 0.0
    race_sums = test_df.groupby('race_id')['syn_win_rate'].transform('sum')
    has_valid_odds = (race_sums > 0) & np.isfinite(race_sums)
    p_norm = np.where(has_valid_odds, test_df['syn_win_rate'] / np.maximum(race_sums, 1e-9), 1.0 / 6.0)
    test_df['init_score'] = probs_to_init_scores(p_norm, clip_eps=1e-5)

    # 3. 両モデル推論 & Gatekeeper 85th% 判定
    print("  [3/4] 両モデル推論 & Gatekeeper スクリーニング中...", flush=True)
    model_honmei = lgb.Booster(model_file=MODEL_HONMEI_PATH)
    feats_honmei = model_honmei.feature_name()
    for f in feats_honmei:
        if f not in test_df.columns: test_df[f] = 0
    test_df['score_honmei'] = model_honmei.predict(test_df[feats_honmei])

    model_residual = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
    feats_residual = model_residual.feature_name()
    for f in feats_residual:
        if f not in test_df.columns: test_df[f] = 0
    raw_res = model_residual.predict(test_df[feats_residual], raw_score=True)
    total_logits = raw_res + test_df['init_score'].to_numpy()
    p_raw_res = 1.0 / (1.0 + np.exp(-np.clip(total_logits, -30, 30)))
    test_df['p_raw_res'] = p_raw_res

    calibrator = get_default_calibrator('platt')
    all_top_p1s = []
    race_gatekeeper_map = {}

    for rid, grp in test_df.groupby('race_id', sort=False):
        s_dict = dict(zip(grp['boat_number'], grp['score_honmei']))
        p_dict_h = calibrator.calibrate_scores(s_dict)
        top_p1 = max(p_dict_h.values())
        all_top_p1s.append(top_p1)
        race_gatekeeper_map[rid] = top_p1

    all_top_p1s = np.array(all_top_p1s)
    dynamic_p1_th = float(np.percentile(all_top_p1s, args.percentile))
    print(f"        -> Gatekeeper 85th% カットオフ閾値: P1 >= {dynamic_p1_th:.4f} ({dynamic_p1_th:.2%})", flush=True)

    # 通過レースのオブジェクトをメモリキャッシュ
    gatekeeper_passed_races = []
    for rid, grp in test_df.groupby('race_id', sort=False):
        if race_gatekeeper_map[rid] < dynamic_p1_th:
            continue
            
        venue_code = race_venue_map.get(rid, int(str(rid).split('_')[0]) if '_' in str(rid) else 1)
        cluster_id = get_venue_cluster(venue_code)
        
        # 残差モデル正規化確率
        p_raw = grp['p_raw_res'].to_numpy(dtype=float)
        p_norm_res = p_raw / np.sum(p_raw)
        
        try:
            r1 = grp[grp['rank'] == 1]['boat_number'].iloc[0]
            r2 = grp[grp['rank'] == 2]['boat_number'].iloc[0]
            r3 = grp[grp['rank'] == 3]['boat_number'].iloc[0]
            actual_combo = f"{int(r1)}-{int(r2)}-{int(r3)}"
        except IndexError:
            actual_combo = None

        gatekeeper_passed_races.append({
            'race_id': rid,
            'venue_code': venue_code,
            'cluster_id': cluster_id,
            'boats': grp['boat_number'].to_numpy(),
            'p_norm_residual': p_norm_res,
            'actual_combo': actual_combo
        })

    print(f"        -> Gatekeeper 通過: {len(gatekeeper_passed_races):,} レースをメモリキャッシュ完了", flush=True)
    c0_count = sum(1 for r in gatekeeper_passed_races if r['cluster_id'] == 0)
    c1_count = sum(1 for r in gatekeeper_passed_races if r['cluster_id'] == 1)
    c2_count = sum(1 for r in gatekeeper_passed_races if r['cluster_id'] == 2)
    print(f"           [Cluster 0: {c0_count} レース, Cluster 1: {c1_count} レース, Cluster 2: {c2_count} レース]", flush=True)

    # 4. Optuna による 6変数同時最適化
    print("\n  [4/4] Optuna による会場クラスタ別 (d2, d3) 最適化開始 (Trials=100)...", flush=True)
    optimizer = PortfolioOptimizer()

    def objective(trial):
        # 6変数サンプリング (0.10 <= d <= 1.20)
        d2_c0 = trial.suggest_float('d2_c0', 0.10, 1.20, step=0.05)
        d3_c0 = trial.suggest_float('d3_c0', 0.10, 1.20, step=0.05)
        
        d2_c1 = trial.suggest_float('d2_c1', 0.10, 1.20, step=0.05)
        d3_c1 = trial.suggest_float('d3_c1', 0.10, 1.20, step=0.05)
        
        d2_c2 = trial.suggest_float('d2_c2', 0.10, 1.20, step=0.05)
        d3_c2 = trial.suggest_float('d3_c2', 0.10, 1.20, step=0.05)

        # 物理制約ペナルティ (d3 < d2 の場合は即時ペナルティ)
        if d3_c0 < d2_c0 or d3_c1 < d2_c1 or d3_c2 < d2_c2:
            return 0.0  # 回収率 0.0% ペナルティ

        cluster_params = {
            0: (d2_c0, d3_c0),
            1: (d2_c1, d3_c1),
            2: (d2_c2, d3_c2)
        }

        res = evaluate_cluster_params(
            cluster_params=cluster_params,
            gatekeeper_passed_races=gatekeeper_passed_races,
            odds_cache=odds_cache,
            optimizer=optimizer,
            bankroll=args.bankroll,
            risk_aversion=args.risk_aversion,
            max_exposure=args.max_exposure,
            max_concentration=args.max_concentration,
            min_ev=args.min_ev,
            max_odds=args.max_odds
        )

        if res['betted'] < 50:
            return 0.0

        trial.set_user_attr('total_profit', res['total_profit'])
        trial.set_user_attr('betted', res['betted'])
        trial.set_user_attr('hits', res['hits'])
        trial.set_user_attr('hit_rate', res['hit_rate'])
        trial.set_user_attr('mdd_amt', res['mdd_amt'])
        
        # 目的関数: 回収率 (ROI) 最大化
        return res['roi']

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    
    # 代表グリッド・ベースライン設定 (d2=0.40, d3=0.60) を初期投入
    study.enqueue_trial({
        'd2_c0': 0.40, 'd3_c0': 0.60,
        'd2_c1': 0.40, 'd3_c1': 0.60,
        'd2_c2': 0.40, 'd3_c2': 0.60
    })

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    best_p = study.best_params
    best_roi = study.best_value

    best_cluster_params = {
        0: (best_p['d2_c0'], best_p['d3_c0']),
        1: (best_p['d2_c1'], best_p['d3_c1']),
        2: (best_p['d2_c2'], best_p['d3_c2'])
    }

    # ベースライン評価 (全クラスタ d2=0.40, d3=0.60)
    baseline_params = {0: (0.40, 0.60), 1: (0.40, 0.60), 2: (0.40, 0.60)}
    eval_base = evaluate_cluster_params(baseline_params, gatekeeper_passed_races, odds_cache, optimizer, args.bankroll, args.risk_aversion, args.max_exposure, args.max_concentration, args.min_ev, args.max_odds)
    eval_best = evaluate_cluster_params(best_cluster_params, gatekeeper_passed_races, odds_cache, optimizer, args.bankroll, args.risk_aversion, args.max_exposure, args.max_concentration, args.min_ev, args.max_odds)

    # 結果出力
    print("\n" + "=" * 75, flush=True)
    print("  📊 会場クラスタ別 Benter パラメーター最適化 結果サマリー", flush=True)
    print("=" * 75, flush=True)
    print(f"  ┌───────────────────────┬────────────────────────┬────────────────────────┐", flush=True)
    print(f"  │ 項目                  │ 一律固定 (d2=0.4, d3=0.6)│ クラスタ別最適化パラメータ │", flush=True)
    print(f"  ├───────────────────────┼────────────────────────┼────────────────────────┤", flush=True)
    print(f"  │ Cluster 0 (イン超強)  │ d2=0.40, d3=0.60       │ d2={best_cluster_params[0][0]:.2f}, d3={best_cluster_params[0][1]:.2f}       │", flush=True)
    print(f"  │ Cluster 1 (難水面)    │ d2=0.40, d3=0.60       │ d2={best_cluster_params[1][0]:.2f}, d3={best_cluster_params[1][1]:.2f}       │", flush=True)
    print(f"  │ Cluster 2 (標準水面)  │ d2=0.40, d3=0.60       │ d2={best_cluster_params[2][0]:.2f}, d3={best_cluster_params[2][1]:.2f}       │", flush=True)
    print(f"  ├───────────────────────┼────────────────────────┼────────────────────────┤", flush=True)
    print(f"  │ 参戦レース数 (Betted) │   {eval_base['betted']:>4d} レース           │   {eval_best['betted']:>4d} レース           │", flush=True)
    print(f"  │ 的中レース数 (的中率) │   {eval_base['hits']:>3d} ({eval_base['hit_rate']:>5.2%})          │   {eval_best['hits']:>3d} ({eval_best['hit_rate']:>5.2%})          │", flush=True)
    print(f"  │ 総投資金額            │  {int(eval_base['total_bet']):>10,d} 円        │  {int(eval_best['total_bet']):>10,d} 円        │", flush=True)
    print(f"  │ 総払戻金額            │  {int(eval_base['total_return']):>10,d} 円        │  {int(eval_best['total_return']):>10,d} 円        │", flush=True)
    print(f"  │ 最終損益 (Total Profit│ {int(eval_base['total_profit']):>+11,d} 円        │ {int(eval_best['total_profit']):>+11,d} 円        │", flush=True)
    print(f"  │ 回収率 (ROI)          │        {eval_base['roi']:>6.2%}          │        {eval_best['roi']:>6.2%}          │", flush=True)
    print(f"  │ 最大ドローダウン(MDD) │  {int(eval_base['mdd_amt']):>10,d} 円        │  {int(eval_best['mdd_amt']):>10,d} 円        │", flush=True)
    print(f"  └───────────────────────┴────────────────────────┴────────────────────────┘", flush=True)

    print(f"\n  🎯 クラスタ別内訳 (最適化後):")
    for c_id, c_name in [(0, "Cluster 0 (イン超強)"), (1, "Cluster 1 (難水面)"), (2, "Cluster 2 (標準水面)")]:
        c_st = eval_best['cluster_stats'][c_id]
        c_roi = (c_st['return_amt'] / c_st['bet_amt']) if c_st['bet_amt'] > 0 else 0.0
        c_hit_rate = (c_st['hits'] / c_st['betted']) if c_st['betted'] > 0 else 0.0
        print(f"    ・{c_name:<20}: 参戦 {c_st['betted']:>3d}R | 的中 {c_st['hits']:>2d}R ({c_hit_rate:.1%}) | 投資 {int(c_st['bet_amt']):>7,d}円 | 払戻 {int(c_st['return_amt']):>7,d}円 | ROI: {c_roi:>6.2%}")

    print(f"\n  ⏱️ 総所要時間: {time.time() - t_start:.2f} 秒\n", flush=True)

    # 設定ファイル保存
    os.makedirs(os.path.dirname(OUTPUT_CONFIG_PATH), exist_ok=True)
    with open(OUTPUT_CONFIG_PATH, 'w', encoding='utf-8') as f:
        json.dump({
            'clusters': {
                'cluster_0': {'name': 'イン超強水面', 'venues': sorted(list(CLUSTER_0_VENUES)), 'd2': best_cluster_params[0][0], 'd3': best_cluster_params[0][1]},
                'cluster_1': {'name': '難水面・イン受難', 'venues': sorted(list(CLUSTER_1_VENUES)), 'd2': best_cluster_params[1][0], 'd3': best_cluster_params[1][1]},
                'cluster_2': {'name': '標準水面', 'venues': 'others', 'd2': best_cluster_params[2][0], 'd3': best_cluster_params[2][1]},
            },
            'best_roi': eval_best['roi'],
            'total_profit': eval_best['total_profit'],
            'mdd_amt': eval_best['mdd_amt'],
            'hit_rate': eval_best['hit_rate'],
            'betted_races': eval_best['betted']
        }, f, indent=4, ensure_ascii=False)
    print(f"  💾 最適クラスタパラメータを保存しました: {OUTPUT_CONFIG_PATH}", flush=True)
    print("=" * 75 + "\n", flush=True)


if __name__ == "__main__":
    main()
