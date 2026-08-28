"""
simulate_betting.py
ボートレース最終アーキテクチャ: Gatekeeper (85th percentile 相対評価) & Extractor 2段階推論パイプライン
- プロセス1 (Gatekeeper): model_honmei.txt (Platt Scaling) の P1 に対し、上位15%分位点 (85th percentile) を動的閾値としてスクリーニング
- プロセス2 (Extractor) : 通過レースに対して model_residual.txt + OddsNormalizer によるオッズ残差高純度確率算出 + 会場クラスタ別 Benter展開 (d2, d3)
- プロセス3 (Optimizer) : portfolio_optimizer.py (EV >= 1.25, Odds <= 30.0, λ=1.0, Max Exp=0.05) による最適資金配分
"""

import os
import time
import math
import argparse
import sqlite3
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb

import train_model
from odds_normalizer import probs_to_init_scores
from probability_calibration import (
    calculate_benter_probs,
    get_default_calibrator,
    load_probability_config,
    load_benter_cluster_config,
    get_cluster_benter_params
)
from portfolio_optimizer import PortfolioOptimizer, load_correlation_mask

# ==========================================
# デフォルト設定 & パス
# ==========================================
MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DB_PATH = 'boatrace.db' if os.path.exists('boatrace.db') else r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'

BANKROLL_DEFAULT = 100000.0     # 初期想定バンクロール (円)
RISK_AVERSION_DEFAULT = 1.0     # リスク回避度 λ
MAX_EXPOSURE_DEFAULT = 0.05     # 1レース最大投資比率 (5% = 5,000円)
MAX_CONCENTRATION_DEFAULT = 0.02 # 1買い目最大投資比率 (2% = 2,000円)
MIN_EV_DEFAULT = 1.25           # 厳格化: 投資対象の最小期待値閾値 (1.25)
MAX_ODDS_DEFAULT = 30.0         # 厳格化: 投資対象の最大オッズ上限 (30.0)
PERCENTILE_DEFAULT = 85.0       # Gatekeeper 上位15% (85th percentile)
KELLY_FRACTION_DEFAULT = 0.25   # Fractional Kelly 係数 (クォーター・ケリー, 最大10%クリップ)


def get_db_connection():
    if os.path.exists('boatrace.db'):
        return sqlite3.connect('boatrace.db')
    return sqlite3.connect(DB_PATH)


def load_all_odds_batch(conn, race_ids):
    """対象レースの3連単オッズデータを一括でメモリキャッシュ"""
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


def run_simulation(
    max_races: int = 5000,
    bankroll: float = BANKROLL_DEFAULT,
    risk_aversion: float = RISK_AVERSION_DEFAULT,
    max_exposure: float = MAX_EXPOSURE_DEFAULT,
    max_concentration: float = MAX_CONCENTRATION_DEFAULT,
    min_ev: float = MIN_EV_DEFAULT,
    max_odds: float = MAX_ODDS_DEFAULT,
    percentile_th: float = PERCENTILE_DEFAULT,
    kelly_fraction: float = KELLY_FRACTION_DEFAULT,
    use_cluster_benter: bool = True,
    model_honmei_path: str = MODEL_HONMEI_PATH,
    model_residual_path: str = MODEL_RESIDUAL_PATH
):
    """
    Gatekeeper (85th percentile) & Extractor (クラスタ別Benter) & Fractional Kelly 最適化バックテスト
    """
    t_start_total = time.time()
    
    # クラスタ別Benter設定ロード
    cluster_cfg = load_benter_cluster_config()

    print("\n" + "=" * 75, flush=True)
    print("  🚀 BOATRACE 最終バックテスト: Gatekeeper (相対評価85th%) & Extractor", flush=True)
    print("=" * 75, flush=True)
    print(f"  Gatekeeper モデル (Honmei) : {model_honmei_path} (Platt Scaling 相対評価)", flush=True)
    print(f"  Extractor モデル (Residual): {model_residual_path} (OddsResidual + 会場クラスタ別Benter)", flush=True)
    print(f"  対象レース数上限           : {max_races:,} レース", flush=True)
    print(f"  Gatekeeper スクリーニング  : 上位 {100.0 - percentile_th:.1f}% 分位点 (85th percentile 動的閾値)", flush=True)
    print(f"  Benter 展開モード          : {'会場クラスタ別動的最適化 (Cluster 0, 1, 2)' if use_cluster_benter else '固定値 (d2=0.40, d3=0.60)'}", flush=True)
    print(f"  Optimizer 厳格化制約       : EV >= {min_ev:.2f}, Odds <= {max_odds:.1f}, λ = {risk_aversion:.1f}", flush=True)
    print(f"  資金配分モデル             : Fractional Kelly (f = {kelly_fraction:.2f}, 動的上限 <= 10.0%)", flush=True)
    print("-" * 75, flush=True)


    # 1. DBから対象Race IDを取得 & オッズ一括ロード
    print("  [1/4] DBから対象レース & 最新オッズデータを一括キャッシュ中...", flush=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY rowid DESC LIMIT ?", [max_races])
    valid_races = [row[0] for row in cursor.fetchall()]
    valid_races_set = set(valid_races)
    print(f"        -> 抽出レース数: {len(valid_races):,} レース", flush=True)
    
    if not valid_races:
        print("❌ エラー: 対象レースIDが取得できませんでした。DB接続を確認してください。")
        conn.close()
        return

    odds_cache = load_all_odds_batch(conn, valid_races)
    conn.close()

    # 2. CSVから該当レースの特徴量データを抽出
    print("  [2/4] CSVからレース特徴量を高速チャンク抽出中...", flush=True)
    t_load_start = time.time()
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=150000):
        matched = chunk[chunk['race_id'].isin(valid_races_set)]
        if len(matched) > 0:
            chunks.append(matched)
            
    if not chunks:
        print("❌ エラー: CSV内に一致するレースが見つかりませんでした。")
        return
        
    df_matched = pd.concat(chunks, ignore_index=True)
    
    # 会場コードマップ事前保存
    race_venue_map = {}
    for rid, vcode in zip(df_matched['race_id'], df_matched['venue_code']):
        if rid not in race_venue_map:
            race_venue_map[rid] = int(vcode)

    print(f"        -> 抽出行数: {len(df_matched):,} 行 (所要時間: {time.time() - t_load_start:.2f}秒)", flush=True)

    # 3. 前処理 & オッズ残差ベースマージン生成
    print("  [3/4] 前処理 & オッズベースマージン (init_score) 算出中...", flush=True)
    test_df = train_model.preprocess_data(df_matched)
    
    if 'syn_win_rate' not in test_df.columns:
        test_df['syn_win_rate'] = 0.0

    race_sums = test_df.groupby('race_id')['syn_win_rate'].transform('sum')
    has_valid_odds = (race_sums > 0) & np.isfinite(race_sums)
    p_norm = np.where(has_valid_odds, test_df['syn_win_rate'] / np.maximum(race_sums, 1e-9), 1.0 / 6.0)
    test_df['init_score'] = probs_to_init_scores(p_norm, clip_eps=1e-5)

    # 4. モデルロード & Gatekeeper 推論による動的閾値算出
    print("  [4/4] Gatekeeper (Honmei) & Extractor (Residual) モデル推論中...", flush=True)
    if not os.path.exists(model_honmei_path) or not os.path.exists(model_residual_path):
        print(f"❌ エラー: モデルファイルが見つかりません。")
        return

    model_honmei = lgb.Booster(model_file=model_honmei_path)
    feats_honmei = model_honmei.feature_name()
    for f in feats_honmei:
        if f not in test_df.columns: test_df[f] = 0

    model_residual = lgb.Booster(model_file=model_residual_path)
    feats_residual = model_residual.feature_name()
    for f in feats_residual:
        if f not in test_df.columns: test_df[f] = 0

    # 本命推論スコアを一括推論
    test_df['score_honmei'] = model_honmei.predict(test_df[feats_honmei])
    calibrator = get_default_calibrator('platt')

    # 全対象レースの Gatekeeper 1着確率 P1 (Top-1) を事前集計
    all_races_p1_top = []
    race_top_p1_map = {}
    
    for rid, group in test_df.groupby('race_id', sort=False):
        s_dict = dict(zip(group['boat_number'], group['score_honmei']))
        p_dict_h = calibrator.calibrate_scores(s_dict)
        top_p1 = max(p_dict_h.values())
        all_races_p1_top.append(top_p1)
        race_top_p1_map[rid] = top_p1

    all_races_p1_top = np.array(all_races_p1_top)
    dynamic_p1_threshold = float(np.percentile(all_races_p1_top, percentile_th))
    
    print(f"\n  🎯 Gatekeeper 相対評価 分位点 (85th percentile) 算出完了:")
    print(f"     ・母集団レース数       : {len(all_races_p1_top):,} レース")
    print(f"     ・P1 最小値 / 平均 / 最大: {all_races_p1_top.min():.2%} / {all_races_p1_top.mean():.2%} / {all_races_p1_top.max():.2%}")
    print(f"     ・動的カットオフ閾値   : P1 >= {dynamic_p1_threshold:.4f} ({dynamic_p1_threshold:.2%})")
    print(f"     ・上位通過予定レース数 : {np.sum(all_races_p1_top >= dynamic_p1_threshold):,} レース ({np.mean(all_races_p1_top >= dynamic_p1_threshold):.2%})")

    # 5. シミュレーション実行 (Gatekeeper通過レースのみ Extractor + Optimizer)
    print("\n  🚀 2段階パイプライン・バックテスト実行中...", flush=True)
    optimizer = PortfolioOptimizer()
    
    race_results_cache = []
    groups = test_df.groupby('race_id', sort=False)
    
    t_sim_start = time.time()
    t_extractor_opt = 0.0
    gatekeeper_passed = 0
    count = 0
    total_target_races = len(groups)
    
    for rid, group in groups:
        count += 1
        top_p1 = race_top_p1_map[rid]
        venue_code = race_venue_map.get(rid, int(str(rid).split('_')[0]) if '_' in str(rid) else 1)
        
        # ----------------------------------------------------
        # プロセス1: Gatekeeper (相対評価 85th percentile 閾値判定)
        # ----------------------------------------------------
        if top_p1 < dynamic_p1_threshold:
            race_results_cache.append({
                'race_id': rid,
                'venue_code': venue_code,
                'cluster_id': 2,
                'cluster_name': 'スキップ',
                'gatekeeper_passed': False,
                'bets': {},
                'actual_combo': None,
                'all_odds': {}
            })
            continue

        gatekeeper_passed += 1
        t0_sub = time.time()

        # ----------------------------------------------------
        # プロセス2: Extractor (model_residual.txt + OddsNormalizer + クラスタ別Benter)
        # ----------------------------------------------------
        raw_res = model_residual.predict(group[feats_residual], raw_score=True)
        total_logits = raw_res + group['init_score'].to_numpy()
        
        p_raw = 1.0 / (1.0 + np.exp(-np.clip(total_logits, -30, 30)))
        p_norm_res = p_raw / np.sum(p_raw)
        p1_dict_res = dict(zip(group['boat_number'], p_norm_res))
        
        # 会場クラスタ別 (d2, d3) パラメーター取得
        if use_cluster_benter:
            d2_c, d3_c, c_id, c_name = get_cluster_benter_params(venue_code, cluster_cfg)
        else:
            d2_c, d3_c, c_id, c_name = 0.40, 0.60, 2, '一律固定'

        # Benterモデルによる120通り3連単確率展開
        benter_probs, _, _ = calculate_benter_probs(
            p1_dict_res,
            d2=d2_c,
            d3=d3_c,
            calibration_method='direct'
        )

        # ----------------------------------------------------
        # プロセス3: Optimizer (厳格制約: EV >= 1.25, Odds <= 30.0)
        # ----------------------------------------------------
        all_odds = odds_cache.get(rid, {})
        if not all_odds:
            bets = {}
        else:
            probs_dict = {p['combo']: p['prob'] for p in benter_probs}
            bets = optimizer.optimize_funds(
                probabilities=probs_dict,
                odds=all_odds,
                bankroll=bankroll,
                risk_aversion=risk_aversion,
                max_exposure=max_exposure,
                max_concentration=max_concentration,
                min_ev=min_ev,
                max_odds=max_odds,
                kelly_fraction=kelly_fraction
            )
            
        t_extractor_opt += (time.time() - t0_sub)
        
        # 実際の結果 (1-2-3着)
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            actual_combo = f"{int(r1)}-{int(r2)}-{int(r3)}"
        except IndexError:
            actual_combo = None
            
        race_results_cache.append({
            'race_id': rid,
            'venue_code': venue_code,
            'cluster_id': c_id,
            'cluster_name': c_name,
            'gatekeeper_passed': True,
            'bets': bets,
            'all_odds': all_odds,
            'actual_combo': actual_combo
        })

    t_sim_elapsed = time.time() - t_sim_start
    t_total_elapsed = time.time() - t_start_total

    # 6. 集計・パフォーマンス評価
    profit_series = []
    betted_profits = []
    stats = {'betted': 0, 'hits': 0, 'bet_amt': 0, 'return_amt': 0}
    cluster_stats = {
        0: {'name': 'Cluster 0 (イン超強)', 'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0},
        1: {'name': 'Cluster 1 (難水面)',   'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0},
        2: {'name': 'Cluster 2 (標準水面)', 'betted': 0, 'hits': 0, 'bet_amt': 0.0, 'return_amt': 0.0}
    }
    
    for r in race_results_cache:
        bets = r['bets']
        if not bets:
            profit_series.append(0)
            continue
            
        bet_sum = sum(bets.values())
        ret_sum = 0
        hit = False
        if r['actual_combo'] and r['actual_combo'] in bets:
            hit = True
            ret_sum = bets[r['actual_combo']] * r['all_odds'].get(r['actual_combo'], 0.0)
            
        stats['betted'] += 1
        stats['bet_amt'] += bet_sum
        stats['return_amt'] += ret_sum
        if hit:
            stats['hits'] += 1
            
        c_id = r['cluster_id']
        if c_id in cluster_stats:
            cluster_stats[c_id]['betted'] += 1
            cluster_stats[c_id]['bet_amt'] += bet_sum
            cluster_stats[c_id]['return_amt'] += ret_sum
            if hit:
                cluster_stats[c_id]['hits'] += 1

        race_profit = ret_sum - bet_sum
        profit_series.append(race_profit)
        betted_profits.append(race_profit)

    # --- 金融工学メトリクスの算出 ---
    cum_pnl = np.cumsum(profit_series)
    peak_pnl = np.maximum.accumulate(cum_pnl)
    drawdown_amt = peak_pnl - cum_pnl
    max_drawdown_amt = float(np.max(drawdown_amt)) if len(drawdown_amt) > 0 else 0.0
    
    equity_series = bankroll + cum_pnl
    peak_equity = np.maximum.accumulate(equity_series)
    pct_drawdown = (peak_equity - equity_series) / np.maximum(peak_equity, 1e-9)
    max_drawdown_pct = float(np.max(pct_drawdown) * 100.0) if len(pct_drawdown) > 0 else 0.0
    
    if len(betted_profits) > 1:
        mean_p = float(np.mean(betted_profits))
        std_p = float(np.std(betted_profits, ddof=1))
        sharpe_per_race = (mean_p / std_p) if std_p > 0 else 0.0
        annualized_sharpe = sharpe_per_race * np.sqrt(len(betted_profits))
    else:
        mean_p = 0.0
        std_p = 0.0
        sharpe_per_race = 0.0
        annualized_sharpe = 0.0

    betted_rate = stats['betted'] / total_target_races if total_target_races > 0 else 0.0
    hit_rate = (stats['hits'] / stats['betted']) if stats['betted'] > 0 else 0.0
    roi = (stats['return_amt'] / stats['bet_amt']) if stats['bet_amt'] > 0 else 0.0
    total_profit = stats['return_amt'] - stats['bet_amt']
    avg_bet = (stats['bet_amt'] / stats['betted']) if stats['betted'] > 0 else 0
    avg_proc_time_ms = (t_sim_elapsed / total_target_races * 1000.0) if total_target_races > 0 else 0.0

    # 7. 最終レポート出力
    print("\n" + "=" * 75, flush=True)
    print("  🏆 バックテスト最終結果サマリー (Gatekeeper 85th% & クラスタ別Benter & 厳格化Optimizer)", flush=True)
    print("=" * 75, flush=True)
    print(f"  総処理レース数          : {total_target_races:,} レース")
    print(f"  Gatekeeper 通過レース   : {gatekeeper_passed:,} レース ({gatekeeper_passed/total_target_races:.2%}) (閾値: P1 >= {dynamic_p1_threshold:.2%})")
    print(f"  参戦レース数 (Betted)   : {stats['betted']:,} レース ({betted_rate:.2%})")
    print(f"  的中レース数 (Hits)     : {stats['hits']:,} レース")
    print(f"  的中率 (Hit Rate)       : {hit_rate:.2%} (全レース基準: {stats['hits']/total_target_races:.2%})")
    print(f"  総投資金額 (Total Bet)  : {int(stats['bet_amt']):,} 円")
    print(f"  総払戻金額 (Return)     : {int(stats['return_amt']):,} 円")
    print(f"  最終損益 (Total Profit) : {int(total_profit):+,} 円")
    print(f"  回収率 (ROI)            : {roi:.2%}")
    print(f"  1レース平均投資額       : {int(avg_bet):,} 円")
    print("-" * 75, flush=True)
    print(f"  想定初期バンクロール    : {int(bankroll):,} 円")
    print(f"  最大ドローダウン (MDD額): {int(max_drawdown_amt):,} 円")
    print(f"  最大ドローダウン (MDD率): {max_drawdown_pct:.2f} %")
    print(f"  1レース平均損益         : {mean_p:+.2f} 円")
    print(f"  損益標準偏差 (Risk)     : {std_p:.2f} 円")
    print(f"  シャープレシオ (PerRace): {sharpe_per_race:.4f}")
    print(f"  サンプル全体シャープレシオ: {annualized_sharpe:.4f}")
    print("-" * 75, flush=True)
    print(f"  🎯 会場クラスタ別内訳:")
    for c_id, c_data in cluster_stats.items():
        c_roi = (c_data['return_amt'] / c_data['bet_amt']) if c_data['bet_amt'] > 0 else 0.0
        c_hit_rate = (c_data['hits'] / c_data['betted']) if c_data['betted'] > 0 else 0.0
        print(f"    ・{c_data['name']:<22}: 参戦 {c_data['betted']:>3d}R | 的中 {c_data['hits']:>2d}R ({c_hit_rate:.1%}) | 投資 {int(c_data['bet_amt']):>7,d}円 | 払戻 {int(c_data['return_amt']):>7,d}円 | ROI: {c_roi:>6.2%}")
    print("-" * 75, flush=True)
    print(f"  1レース平均処理時間     : {avg_proc_time_ms:.3f} ms / レース")
    print(f"  シミュレーション総時間  : {t_sim_elapsed:.2f} 秒")
    print(f"  Extractor+Opt 累計時間  : {t_extractor_opt:.2f} 秒")
    print(f"  全工程総所要時間        : {t_total_elapsed:.2f} 秒")
    print("=" * 75 + "\n", flush=True)

    return {
        'total_races': total_target_races,
        'gatekeeper_passed': gatekeeper_passed,
        'dynamic_threshold': dynamic_p1_threshold,
        'betted_races': stats['betted'],
        'hits': stats['hits'],
        'hit_rate': hit_rate,
        'total_bet': stats['bet_amt'],
        'total_return': stats['return_amt'],
        'total_profit': total_profit,
        'roi': roi,
        'mdd_amt': max_drawdown_amt,
        'mdd_pct': max_drawdown_pct,
        'sharpe': sharpe_per_race,
        'cluster_stats': cluster_stats,
        'avg_proc_time_ms': avg_proc_time_ms
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulate Betting with Gatekeeper 85th Percentile & Cluster Benter & Strict Optimizer")
    parser.add_argument('--races', type=int, default=5000, help="Number of races to simulate (default: 5000)")
    parser.add_argument('--bankroll', type=float, default=BANKROLL_DEFAULT, help="Bankroll in JPY (default: 100,000)")
    parser.add_argument('--risk_aversion', type=float, default=RISK_AVERSION_DEFAULT, help="Risk aversion lambda (default: 1.0)")
    parser.add_argument('--max_exposure', type=float, default=MAX_EXPOSURE_DEFAULT, help="Max exposure per race (default: 0.05)")
    parser.add_argument('--max_concentration', type=float, default=MAX_CONCENTRATION_DEFAULT, help="Max concentration per combo (default: 0.02)")
    parser.add_argument('--min_ev', type=float, default=MIN_EV_DEFAULT, help="Minimum EV threshold (default: 1.25)")
    parser.add_argument('--max_odds', type=float, default=MAX_ODDS_DEFAULT, help="Maximum Odds upper bound (default: 30.0)")
    parser.add_argument('--percentile', type=float, default=PERCENTILE_DEFAULT, help="Gatekeeper percentile cutoff (default: 85.0)")
    parser.add_argument('--kelly_fraction', type=float, default=KELLY_FRACTION_DEFAULT, help="Fractional Kelly fraction (default: 0.25)")
    parser.add_argument('--use_cluster_benter', action='store_true', default=True, help="Use cluster-specific Benter parameters")
    parser.add_argument('--model_honmei', type=str, default=MODEL_HONMEI_PATH, help="Path to Gatekeeper honmei model")
    parser.add_argument('--model_residual', type=str, default=MODEL_RESIDUAL_PATH, help="Path to Extractor residual model")
    
    args = parser.parse_args()
    run_simulation(
        max_races=args.races,
        bankroll=args.bankroll,
        risk_aversion=args.risk_aversion,
        max_exposure=args.max_exposure,
        max_concentration=args.max_concentration,
        min_ev=args.min_ev,
        max_odds=args.max_odds,
        percentile_th=args.percentile,
        kelly_fraction=args.kelly_fraction,
        use_cluster_benter=args.use_cluster_benter,
        model_honmei_path=args.model_honmei,
        model_residual_path=args.model_residual
    )

