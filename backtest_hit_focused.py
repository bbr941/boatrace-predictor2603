"""
backtest_hit_focused.py
🚤 的中特化（動的ダッチング・トリガミ回避）モード専用 バックテスト検証スクリプト

【検証項目】
1. 基本スペック: 総レース数、的中レース数、的中率(%)、総投資額、総回収額、全体回収率(ROI %)、純損益
2. コロガシ適性: 連勝数分布（1連勝、2連勝、3連勝、4連勝、5連勝... 最大連勝数）
3. 連敗分布: 最大連敗数、平均連敗数
4. 月別 & Gatekeeper P1帯別 & 買い目点数別の詳細ブレークダウン
"""

import os
import sys
import time
import math
import sqlite3
import argparse
from typing import Dict, List, Tuple, Any
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import lightgbm as lgb

# プロジェクトルート
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import train_model
from odds_normalizer import probs_to_init_scores
from probability_calibration import (
    calculate_benter_probs,
    get_default_calibrator,
    load_benter_cluster_config,
    get_cluster_benter_params
)
from portfolio_optimizer import calculate_dutching_bets

# デフォルト定数
DEFAULT_DATA_PATH = 'train_data_full.csv'
DEFAULT_DB_PATH = 'boatrace.db'
MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'
DEFAULT_START_DATE = '2026-01-01'
DEFAULT_BUDGET = 1000
DEFAULT_TARGET_CUM_PROB = 0.50
DEFAULT_MAX_COMBOS = 8
DEFAULT_MIN_COMBOS = 2


def run_backtest(
    data_path: str = DEFAULT_DATA_PATH,
    db_path: str = DEFAULT_DB_PATH,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = None,
    budget: int = DEFAULT_BUDGET,
    target_cum_prob: float = DEFAULT_TARGET_CUM_PROB,
    max_combos: int = DEFAULT_MAX_COMBOS,
    min_combos: int = DEFAULT_MIN_COMBOS,
    sample_limit: int = None
):
    print("=" * 85, flush=True)
    print(" 🎯 [BACKTEST] 的中特化（動的ダッチング）モード パフォーマンス & 連勝分布検証", flush=True)
    print("=" * 85, flush=True)
    print(f"  ・対象データセット : {data_path}", flush=True)
    print(f"  ・検証対象期間     : {start_date} 〜 {end_date if end_date else '最新'}", flush=True)
    print(f"  ・1レース投資予算  : {budget:,} 円 (オッズ逆数比ダッチング配分)", flush=True)
    print(f"  ・抽出累積確率     : 上位 {target_cum_prob:.0%} (最大 {max_combos} 点 / 最小 {min_combos} 点)", flush=True)
    print("-" * 85, flush=True)

    t_start = time.time()

    # 1. データセット読み込み
    print(f"\n[1/5] データセット読み込み中 ({data_path})...", end=" ", flush=True)
    t0 = time.time()
    df_all = pd.read_csv(data_path)
    
    # 期間フィルタリング
    cond = df_all['race_date'] >= start_date
    if end_date:
        cond &= df_all['race_date'] <= end_date
    df_test = df_all[cond].copy()
    
    if sample_limit and sample_limit > 0:
        unique_races = df_test['race_id'].unique()[:sample_limit]
        df_test = df_test[df_test['race_id'].isin(unique_races)].copy()
        
    num_races = df_test['race_id'].nunique()
    print(f"完了! ({len(df_test):,}行 / {num_races:,}レース, {time.time()-t0:.2f}秒)", flush=True)

    if num_races == 0:
        print("❌ 対象レースが存在しませんでした。期間やデータパスを確認してください。")
        return

    # 2. 確定着順 (1-2-3着) & レースメタ情報の抽出
    print("[2/5] 確定着順 (1-3着) & 会場・日付メタ情報を抽出中...", end=" ", flush=True)
    t0 = time.time()
    
    # 確定3連単コンボ
    df_top3 = df_test[df_test['rank'].isin([1, 2, 3])].sort_values(['race_id', 'rank'])
    actual_combos = (
        df_top3.groupby('race_id')['boat_number']
        .apply(lambda s: '-'.join(map(str, s.astype(int))))
        .to_dict()
    )
    
    # レースごとのメタデータ (race_date, venue_code)
    race_meta = (
        df_test.groupby('race_id')
        .agg({
            'race_date': 'first',
            'venue_code': 'first'
        })
        .to_dict(orient='index')
    )
    print(f"完了! ({time.time()-t0:.2f}秒)", flush=True)

    # 3. 前処理 & モデル推論 (本命 & 残差)
    print("[3/5] 前処理 & 本命・残差モデルの一括ベクトル推論中...", end=" ", flush=True)
    t0 = time.time()
    test_df = train_model.preprocess_data(df_test)
    if 'syn_win_rate' not in test_df.columns:
        test_df['syn_win_rate'] = 0.0

    race_sums = test_df.groupby('race_id')['syn_win_rate'].transform('sum')
    has_valid_odds = (race_sums > 0) & np.isfinite(race_sums)
    p_norm = np.where(has_valid_odds, test_df['syn_win_rate'] / np.maximum(race_sums, 1e-9), 1.0 / 6.0)
    test_df['init_score'] = probs_to_init_scores(p_norm, clip_eps=1e-5)

    # モデルロード
    model_honmei = lgb.Booster(model_file=MODEL_HONMEI_PATH)
    feats_honmei = model_honmei.feature_name()
    for f in feats_honmei:
        if f not in test_df.columns: test_df[f] = 0

    model_residual = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
    feats_residual = model_residual.feature_name()
    for f in feats_residual:
        if f not in test_df.columns: test_df[f] = 0

    # 一括推論
    test_df['score_honmei'] = model_honmei.predict(test_df[feats_honmei])
    test_df['raw_res'] = model_residual.predict(test_df[feats_residual], raw_score=True)
    test_df['total_logits'] = test_df['raw_res'] + test_df['init_score'].to_numpy()
    
    p_raw = 1.0 / (1.0 + np.exp(-np.clip(test_df['total_logits'].to_numpy(), -30, 30)))
    test_df['p_raw'] = p_raw
    race_p_sums = test_df.groupby('race_id')['p_raw'].transform('sum')
    test_df['p_norm_res'] = test_df['p_raw'] / np.maximum(race_p_sums, 1e-9)

    calibrator = get_default_calibrator('platt')
    cluster_cfg = load_benter_cluster_config()
    cluster_benter_lookup = {v: get_cluster_benter_params(v, cluster_cfg) for v in range(1, 25)}

    # レースごとの残差確率およびGatekeeper P1抽出
    race_p1_res_dict = {}
    race_gk_p1_dict = {}

    for rid, group in test_df.groupby('race_id', sort=False):
        s_dict = dict(zip(group['boat_number'], group['score_honmei']))
        p_dict_h = calibrator.calibrate_scores(s_dict)
        race_gk_p1_dict[rid] = max(p_dict_h.values()) if p_dict_h else 0.0
        race_p1_res_dict[rid] = dict(zip(group['boat_number'], group['p_norm_res']))

    print(f"完了! ({time.time()-t0:.2f}秒)", flush=True)

    # 4. DBからオッズデータを一括ロード
    print("[4/5] SQLite DBから直前オッズデータを一括ロード中...", end=" ", flush=True)
    t0 = time.time()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    odds_cache = {}
    
    all_rids = list(race_p1_res_dict.keys())
    chunk_size = 1000
    for i in range(0, len(all_rids), chunk_size):
        chunk = all_rids[i:i+chunk_size]
        ph = ','.join(['?'] * len(chunk))
        cursor.execute(f"SELECT race_id, combination, odds_1min FROM odds_data WHERE race_id IN ({ph})", chunk)
        for rid, combo, val in cursor.fetchall():
            c_str = str(combo)
            if len(c_str) == 3:
                fmt_comb = f"{c_str[0]}-{c_str[1]}-{c_str[2]}"
                if rid not in odds_cache: odds_cache[rid] = {}
                odds_cache[rid][fmt_comb] = float(val) if val else 0.0
    conn.close()
    print(f"完了! ({len(odds_cache):,} レース取得, {time.time()-t0:.2f}秒)", flush=True)

    # 5. 全レースの的中特化ダッチング計算 & 成績シミュレーション
    print("[5/5] 全レースの的中特化（動的ダッチング）シミュレーションを実行中...", end=" ", flush=True)
    t0 = time.time()
    
    simulation_results = []
    
    for rid in all_rids:
        all_odds = odds_cache.get(rid, {})
        if not all_odds:
            continue
            
        actual_combo = actual_combos.get(rid)
        if not actual_combo:
            continue
            
        meta = race_meta.get(rid, {})
        r_date = meta.get('race_date', '')
        v_code = int(meta.get('venue_code', 1))
        gk_p1 = race_gk_p1_dict.get(rid, 0.0)
        
        d2_c, d3_c, c_id, c_name = cluster_benter_lookup.get(v_code, (0.40, 0.60, 2, '標準水面'))
        p1_res = race_p1_res_dict[rid]
        
        benter_probs, _, _ = calculate_benter_probs(
            p1_res,
            d2=d2_c,
            d3=d3_c,
            calibration_method='direct'
        )
        benter_probs_dict = {p['combo']: p['prob'] for p in benter_probs}
        
        # 動的ダッチング配分
        bets = calculate_dutching_bets(
            benter_probs=benter_probs_dict,
            odds_dict=all_odds,
            budget=budget,
            target_cum_prob=target_cum_prob,
            max_combos=max_combos,
            min_combos=min_combos
        )
        
        if not bets:
            continue
            
        total_invest = sum(bets.values())
        is_hit = (actual_combo in bets)
        
        if is_hit:
            hit_odds = all_odds.get(actual_combo, 0.0)
            hit_bet = bets[actual_combo]
            payout = int((hit_bet / 100.0) * hit_odds * 100.0)
        else:
            hit_odds = 0.0
            hit_bet = 0
            payout = 0
            
        profit = payout - total_invest
        
        # 累積確率の算出
        cum_prob_bought = sum(benter_probs_dict.get(c, 0.0) for c in bets)
        
        simulation_results.append({
            'race_id': rid,
            'race_date': r_date,
            'venue_code': v_code,
            'cluster_id': c_id,
            'cluster_name': c_name,
            'gk_p1': gk_p1,
            'actual_combo': actual_combo,
            'bets_count': len(bets),
            'cum_prob': cum_prob_bought,
            'total_invest': total_invest,
            'is_hit': is_hit,
            'hit_odds': hit_odds,
            'payout': payout,
            'profit': profit
        })

    print(f"完了! ({len(simulation_results):,} レース評価, {time.time()-t0:.2f}秒)\n", flush=True)

    if not simulation_results:
        print("❌ 有効なシミュレーション結果が得られませんでした。")
        return

    df_res = pd.DataFrame(simulation_results)
    
    # 時系列順（日付・レースID順）にソート
    df_res.sort_values(by=['race_date', 'race_id'], inplace=True)
    df_res.reset_index(drop=True, inplace=True)

    # =====================================================================
    # 統計・パフォーマンス集計
    # =====================================================================
    total_eval_races = len(df_res)
    total_hits = df_res['is_hit'].sum()
    hit_rate = total_hits / total_eval_races if total_eval_races > 0 else 0.0
    
    total_invest_sum = df_res['total_invest'].sum()
    total_payout_sum = df_res['payout'].sum()
    total_net_profit = total_payout_sum - total_invest_sum
    roi = (total_payout_sum / total_invest_sum * 100.0) if total_invest_sum > 0 else 0.0

    avg_invest = df_res['total_invest'].mean()
    avg_payout_when_hit = df_res[df_res['is_hit']]['payout'].mean() if total_hits > 0 else 0
    avg_profit_when_hit = df_res[df_res['is_hit']]['profit'].mean() if total_hits > 0 else 0
    min_profit_when_hit = df_res[df_res['is_hit']]['profit'].min() if total_hits > 0 else 0
    avg_combos = df_res['bets_count'].mean()
    avg_cum_prob = df_res['cum_prob'].mean()

    # =====================================================================
    # 連勝・連敗分布（コロガシ適性）の算出
    # =====================================================================
    hit_streaks = []
    loss_streaks = []
    
    curr_hit_streak = 0
    curr_loss_streak = 0
    
    for h in df_res['is_hit']:
        if h:
            curr_hit_streak += 1
            if curr_loss_streak > 0:
                loss_streaks.append(curr_loss_streak)
                curr_loss_streak = 0
        else:
            curr_loss_streak += 1
            if curr_hit_streak > 0:
                hit_streaks.append(curr_hit_streak)
                curr_hit_streak = 0
                
    if curr_hit_streak > 0:
        hit_streaks.append(curr_hit_streak)
    if curr_loss_streak > 0:
        loss_streaks.append(curr_loss_streak)

    hit_streak_counts = Counter(hit_streaks)
    loss_streak_counts = Counter(loss_streaks)
    max_hit_streak = max(hit_streaks) if hit_streaks else 0
    max_loss_streak = max(loss_streaks) if loss_streaks else 0
    avg_hit_streak = np.mean(hit_streaks) if hit_streaks else 0
    avg_loss_streak = np.mean(loss_streaks) if loss_streaks else 0

    # =====================================================================
    # レポート出力
    # =====================================================================
    print("=" * 85)
    print(" 📊 【的中特化（動的ダッチング）バックテスト 総合パフォーマンス結果】")
    print("=" * 85)
    print(f"  ・検証レース総数       : {total_eval_races:,} レース")
    print(f"  ・的中レース数 (Hits)  : {total_hits:,} レース")
    print(f"  ・ベース的中率 (Hit %) : {hit_rate:.2%} ({total_hits}/{total_eval_races})")
    print(f"  ・平均買い目点数       : {avg_combos:.2f} 点 (平均累積勝率: {avg_cum_prob:.1%})")
    print("-" * 85)
    print(f"  ・総投資金額 (Total)   : {total_invest_sum:,} 円 (平均 {avg_invest:.0f}円/R)")
    print(f"  ・総払戻金額 (Payout)  : {total_payout_sum:,} 円")
    print(f"  ・確定純損益 (Profit)  : {total_net_profit:+,} 円")
    print(f"  ・全体回収率 (ROI)     : {roi:.2f} %")
    print("-" * 85)
    print(f"  ・的中時 平均払戻金    : {avg_payout_when_hit:,.0f} 円")
    print(f"  ・的中時 平均純利益    : {avg_profit_when_hit:+,.0f} 円")
    print(f"  ・的中時 最低純利益    : {min_profit_when_hit:+,.0f} 円 (※トリガミ完全回避の確認)")
    print("=" * 85)

    print("\n" + "=" * 85)
    print(" 🎲 【連勝分布（コロガシ・継続適性分析）】")
    print("=" * 85)
    print(f"  ・最大連勝数 (Max Hit Streak)   : {max_hit_streak} 連勝")
    print(f"  ・平均連勝数 (Avg Hit Streak)   : {avg_hit_streak:.2f} 連勝")
    print(f"  ・的中セッション総数            : {len(hit_streaks):,} 回")
    print("-" * 85)
    print("  【連勝数ごとの発生頻度（何連勝でストップしたか）】")
    
    # 1連勝から順に表示
    for s in range(1, max(max_hit_streak + 1, 10)):
        cnt = hit_streak_counts.get(s, 0)
        pct = (cnt / len(hit_streaks) * 100.0) if hit_streaks else 0.0
        # 累積で s 連勝以上達成した確率
        cum_s_cnt = sum(c for k, c in hit_streak_counts.items() if k >= s)
        cum_s_pct = (cum_s_cnt / len(hit_streaks) * 100.0) if hit_streaks else 0.0
        bar = "█" * int(round(pct / 2.5))
        if cnt > 0 or s <= 7:
            print(f"    {s:2d} 連勝 : {cnt:5,d} 回 ({pct:5.1f}%) | {s}連勝以上達成率: {cum_s_pct:5.1f}% | {bar}")

    print("-" * 85)
    print(" ⚠️ 【連敗分布（ドローダウン耐性）】")
    print(f"  ・最大連敗数 (Max Loss Streak) : {max_loss_streak} 連続不的中")
    print(f"  ・平均連敗数 (Avg Loss Streak) : {avg_loss_streak:.2f} 連続不的中")
    print(f"  ・不的中セッション総数         : {len(loss_streaks):,} 回")
    print("  【主な連敗数発生頻度】")
    for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        cnt = loss_streak_counts.get(s, 0)
        pct = (cnt / len(loss_streaks) * 100.0) if loss_streaks else 0.0
        if cnt > 0:
            print(f"    {s:2d} 連敗 : {cnt:5,d} 回 ({pct:5.1f}%)")
    print("=" * 85)

    # =====================================================================
    # Gatekeeper P1 帯別のパフォーマンス内訳
    # =====================================================================
    print("\n" + "=" * 85)
    print(" 🛡️ 【Gatekeeper P1 帯別 パフォーマンス内訳】")
    print("=" * 85)
    
    df_res['p1_tier'] = pd.cut(
        df_res['gk_p1'],
        bins=[0.0, 0.50, 0.65, 0.7438, 0.85, 1.00],
        labels=['< 50% (波乱)', '50-65% (混戦)', '65-74.38% (有力)', '74.38-85% (本命堅)', '>= 85% (超鉄板)']
    )
    
    tier_summary = df_res.groupby('p1_tier', observed=False).agg(
        races=('race_id', 'count'),
        hits=('is_hit', 'sum'),
        invest=('total_invest', 'sum'),
        payout=('payout', 'sum')
    )
    tier_summary['hit_rate'] = tier_summary['hits'] / np.maximum(tier_summary['races'], 1)
    tier_summary['roi'] = tier_summary['payout'] / np.maximum(tier_summary['invest'], 1) * 100.0
    tier_summary['profit'] = tier_summary['payout'] - tier_summary['invest']
    
    print(f"{'P1 帯域':<20} | {'レース数':>8} | {'的中数':>8} | {'的中率':>8} | {'総投資 (円)':>12} | {'総払戻 (円)':>12} | {'回収率 (ROI)':>10} | {'損益 (円)':>12}")
    print("-" * 105)
    for tier, row in tier_summary.iterrows():
        r_cnt = int(row['races'])
        h_cnt = int(row['hits'])
        inv = int(row['invest'])
        pay = int(row['payout'])
        prof = int(row['profit'])
        print(f"{str(tier):<20} | {r_cnt:8,d} | {h_cnt:8,d} | {row['hit_rate']:7.2%} | {inv:12,d} | {pay:12,d} | {row['roi']:9.2f}% | {prof:+12,d}")
    print("=" * 85)

    # =====================================================================
    # 月別推移 (Monthly Trend)
    # =====================================================================
    print("\n" + "=" * 85)
    print(" 📅 【月別 パフォーマンス推移】")
    print("=" * 85)
    
    df_res['month'] = df_res['race_date'].str[:7]
    month_summary = df_res.groupby('month').agg(
        races=('race_id', 'count'),
        hits=('is_hit', 'sum'),
        invest=('total_invest', 'sum'),
        payout=('payout', 'sum')
    )
    month_summary['hit_rate'] = month_summary['hits'] / np.maximum(month_summary['races'], 1)
    month_summary['roi'] = month_summary['payout'] / np.maximum(month_summary['invest'], 1) * 100.0
    month_summary['profit'] = month_summary['payout'] - month_summary['invest']
    
    print(f"{'年月':<10} | {'レース数':>8} | {'的中数':>8} | {'的中率':>8} | {'総投資 (円)':>12} | {'総払戻 (円)':>12} | {'回収率 (ROI)':>10} | {'損益 (円)':>12}")
    print("-" * 95)
    for m, row in month_summary.iterrows():
        r_cnt = int(row['races'])
        h_cnt = int(row['hits'])
        inv = int(row['invest'])
        pay = int(row['payout'])
        prof = int(row['profit'])
        print(f"{str(m):<10} | {r_cnt:8,d} | {h_cnt:8,d} | {row['hit_rate']:7.2%} | {inv:12,d} | {pay:12,d} | {row['roi']:9.2f}% | {prof:+12,d}")
    print("=" * 85)

    # =====================================================================
    # コロガシ（2連・3連）適性シミュレーション
    # =====================================================================
    print("\n" + "=" * 85)
    print(" 🌀 【コロガシ（均等分散転がし）適性シミュレーション】")
    print("=" * 85)
    
    # 2連コロガシ (2レース連続的中)
    # 3連コロガシ (3レース連続的中)
    total_2_step_trials = max(total_eval_races - 1, 1)
    hits_array = df_res['is_hit'].to_numpy()
    success_2_step = sum((hits_array[i] and hits_array[i+1]) for i in range(len(hits_array)-1))
    rate_2_step = success_2_step / total_2_step_trials if total_2_step_trials > 0 else 0.0
    
    total_3_step_trials = max(total_eval_races - 2, 1)
    success_3_step = sum((hits_array[i] and hits_array[i+1] and hits_array[i+2]) for i in range(len(hits_array)-2))
    rate_3_step = success_3_step / total_3_step_trials if total_3_step_trials > 0 else 0.0
    
    print(f"  ・2レース連続的中 (2連コロガシ) 成功回数 : {success_2_step:,} / {total_2_step_trials:,} 試行 ({rate_2_step:.2%})")
    print(f"  ・3レース連続的中 (3連コロガシ) 成功回数 : {success_3_step:,} / {total_3_step_trials:,} 試行 ({rate_3_step:.2%})")
    print("=" * 85)

    elapsed_total = time.time() - t_start
    print(f"\n✨ バックテスト完了 (総処理時間: {elapsed_total:.2f}秒)\n", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="的中特化モード バックテスト")
    parser.add_argument('--data-path', type=str, default=DEFAULT_DATA_PATH, help='データセットCSVパス')
    parser.add_argument('--db-path', type=str, default=DEFAULT_DB_PATH, help='SQLite DBパス')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE, help='検証開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='検証終了日 (YYYY-MM-DD)')
    parser.add_argument('--budget', type=int, default=DEFAULT_BUDGET, help='1レースあたり予算 (円)')
    parser.add_argument('--target-cum-prob', type=float, default=DEFAULT_TARGET_CUM_PROB, help='目標累積勝率 (0.50)')
    parser.add_argument('--max-combos', type=int, default=DEFAULT_MAX_COMBOS, help='最大買い目数 (8)')
    parser.add_argument('--min-combos', type=int, default=DEFAULT_MIN_COMBOS, help='最小買い目数 (2)')
    parser.add_argument('--sample-limit', type=int, default=None, help='検証レース数の上限 (テスト用)')

    args = parser.parse_args()

    run_backtest(
        data_path=args.data_path,
        db_path=args.db_path,
        start_date=args.start_date,
        end_date=args.end_date,
        budget=args.budget,
        target_cum_prob=args.target_cum_prob,
        max_combos=args.max_combos,
        min_combos=args.min_combos,
        sample_limit=args.sample_limit
    )
