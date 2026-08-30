"""
analyze_micro_clusters.py
過去データ（2026年1月〜8月 Out-of-Time テストデータ）から高回収率の条件（マイクロクラスタ）を
決定木（DecisionTreeRegressor: Sniper Approach）により事後抽出・可視化する分析スクリプト
"""

import os
import sys
import time
import math
import sqlite3
import itertools
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor, export_text

# プロジェクトルートのパス追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import train_model
from odds_normalizer import probs_to_init_scores
from probability_calibration import (
    calculate_benter_probs,
    get_default_calibrator,
    load_benter_cluster_config,
    get_cluster_benter_params
)

# =====================================================================
# 設定 & パラメータ
# =====================================================================
DATA_PATH = 'train_data_full.csv'
DB_PATH = 'boatrace.db' if os.path.exists('boatrace.db') else r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'

TEST_START_DATE = '2026-01-01'
GATEKEEPER_TH = 0.7438     # Gatekeeper 黄金ベースライン P1 閾値
MIN_EV = 1.25              # ベット候補最小EV
MAX_ODDS = 30.0            # ベット候補最大オッズ上限

def run_micro_cluster_analysis():
    t_start = time.time()
    print("=" * 85)
    print(" 🌲 [MICRO-CLUSTERING ANALYSIS] 決定木による高回収率マイクロクラスタ抽出 (Sniper)")
    print("=" * 85)
    print(f"  ・対象テスト期間 : {TEST_START_DATE} 以降 (Out-of-Time 未見データ)")
    print(f"  ・Gatekeeper P1 : >= {GATEKEEPER_TH:.4f} ({GATEKEEPER_TH:.2%})")
    print(f"  ・抽出条件       : EV >= {MIN_EV:.2f}, Odds <= {MAX_ODDS:.1f}")
    print("-" * 85)

    # 1. データセット読み込み
    print(f"\n[1/5] データセット ({DATA_PATH}) から {TEST_START_DATE} 以降のテストデータを抽出中...", flush=True)
    t0 = time.time()
    df_all = pd.read_csv(DATA_PATH)
    df_test = df_all[df_all['race_date'] >= TEST_START_DATE].copy()
    print(f"      -> 抽出行数: {len(df_test):,} 行 | レース数: {df_test['race_id'].nunique():,} レース ({time.time()-t0:.2f}秒)")

    # レースごとのメタ情報を高速抽出（辞書化）
    print("[2/5] レース単位のメタ情報 & 確定結果を高速抽出中...", flush=True)
    t0 = time.time()
    
    # 1号艇データのみ抽出してマージ
    df_b1 = df_test[df_test['boat_number'] == 1].drop_duplicates('race_id').set_index('race_id')
    
    # 確定着順 (1, 2, 3着) のベクトル抽出
    df_top3 = df_test[df_test['rank'].isin([1, 2, 3])].sort_values(['race_id', 'rank'])
    actual_combos = (
        df_top3.groupby('race_id')['boat_number']
        .apply(lambda s: '-'.join(map(str, s.astype(int))))
        .to_dict()
    )

    print(f"      -> メタ情報キャッシュ完了 ({time.time()-t0:.2f}秒)")

    # 2. 前処理 & モデル一括推論
    print("[3/5] 前処理 & 本命/残差モデルの一括ベクトル推論中...", flush=True)
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

    # 一括推論（超高速）
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

    print(f"      -> 推論完了 ({time.time()-t0:.2f}秒)")

    # 3. Gatekeeper スクリーニング & オッズキャッシュ
    print("[4/5] Gatekeeper フィルタリング & Benter 120通り確率展開中...", flush=True)
    t0 = time.time()
    
    # レースごとにGatekeeper P1判定
    passed_race_ids = []
    race_p1_dict = {}
    race_p_gap_dict = {}
    race_p1_res_dict = {}

    for rid, group in test_df.groupby('race_id', sort=False):
        s_dict = dict(zip(group['boat_number'], group['score_honmei']))
        p_dict_h = calibrator.calibrate_scores(s_dict)
        
        sorted_p = sorted(p_dict_h.values(), reverse=True)
        top_p1 = sorted_p[0]
        prob_gap = sorted_p[0] - sorted_p[1]
        
        if top_p1 >= GATEKEEPER_TH:
            passed_race_ids.append(rid)
            race_p1_dict[rid] = top_p1
            race_p_gap_dict[rid] = prob_gap
            race_p1_res_dict[rid] = dict(zip(group['boat_number'], group['p_norm_res']))

    print(f"      -> Gatekeeper 通過: {len(passed_race_ids):,} / {test_df['race_id'].nunique():,} レース ({len(passed_race_ids)/test_df['race_id'].nunique():.2%})")

    # DBから通過レースのオッズデータを一括ロード
    print("      -> DBからオッズデータを一括ロード中...", end=" ", flush=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    odds_cache = {}
    chunk_size = 500
    for i in range(0, len(passed_race_ids), chunk_size):
        chunk = passed_race_ids[i:i+chunk_size]
        ph = ','.join(['?']*len(chunk))
        cursor.execute(f"SELECT race_id, combination, odds_1min FROM odds_data WHERE race_id IN ({ph})", chunk)
        for rid, combo, val in cursor.fetchall():
            c_str = str(combo)
            if len(c_str) == 3:
                fmt_comb = f"{c_str[0]}-{c_str[1]}-{c_str[2]}"
                if rid not in odds_cache: odds_cache[rid] = {}
                odds_cache[rid][fmt_comb] = float(val) if val else 0.0
    conn.close()
    print(f"完了 ({len(odds_cache):,} レースキャッシュ)")

    # 4. ベット候補（EV >= 1.25, Odds <= 30.0）の抽出と損益計算
    bet_records = []
    
    for rid in passed_race_ids:
        all_odds = odds_cache.get(rid, {})
        if not all_odds:
            continue
            
        p1_res = race_p1_res_dict[rid]
        
        # 1号艇メタデータ
        if rid in df_b1.index:
            b1_row = df_b1.loc[rid]
            vcode = int(b1_row.get('venue_code', 1))
            w_speed = float(b1_row.get('wind_speed', 0.0))
            w_height = float(b1_row.get('wave_height', 0.0))
            b1_motor = float(b1_row.get('motor_rate', 30.0))
            b1_boat = float(b1_row.get('boat_rate', 30.0))
            b1_nat_win = float(b1_row.get('nat_win_rate', 5.0))
            b1_local_win = float(b1_row.get('local_win_rate', 5.0))
            b1_weight = float(b1_row.get('weight', 52.0))
            b1_mom_diff = float(b1_row.get('ex_momentum_diff', 0.0))
            b1_diff_min = float(b1_row.get('ex_diff_from_race_min', 0.0))
            b1_mom_dev = float(b1_row.get('ex_momentum_deviation', 0.0))
            w_nige_vuln = float(b1_row.get('wind_nige_vulnerability', 0.0))
            w_makuri_cr = float(b1_row.get('wind_makuri_cross', 0.0))
            hw_risk = float(b1_row.get('high_wave_inner_risk', 0.0))
            wave_w_prod = float(b1_row.get('wave_weight_prod', 0.0))
        else:
            vcode = 1
            w_speed, w_height, b1_motor, b1_boat, b1_nat_win, b1_local_win, b1_weight = 0, 0, 30, 30, 5, 5, 52
            b1_mom_diff, b1_diff_min, b1_mom_dev, w_nige_vuln, w_makuri_cr, hw_risk, wave_w_prod = 0, 0, 0, 0, 0, 0, 0

        d2_c, d3_c, c_id, c_name = cluster_benter_lookup.get(vcode, (0.40, 0.60, 2, '標準水面'))
        benter_probs, _, _ = calculate_benter_probs(p1_res, d2=d2_c, d3=d3_c, calibration_method='direct')
        
        actual_combo = actual_combos.get(rid)
        top_p1 = race_p1_dict[rid]
        prob_gap = race_p_gap_dict[rid]
        
        for bp in benter_probs:
            combo = bp['combo']
            prob = bp['prob']
            odds = all_odds.get(combo, 0.0)
            if odds <= 0 or odds > MAX_ODDS:
                continue
                
            ev = prob * odds
            if ev >= MIN_EV:
                is_hit = (combo == actual_combo)
                bet_amt = 100.0
                payout = (odds * 100.0) if is_hit else 0.0
                profit = payout - bet_amt
                ret_rate = payout / bet_amt
                
                c_parts = [int(x) for x in combo.split('-')]
                
                bet_records.append({
                    'race_id': rid,
                    'combo': combo,
                    'head_boat': c_parts[0],
                    'second_boat': c_parts[1],
                    'third_boat': c_parts[2],
                    'prob': prob,
                    'odds': odds,
                    'ev': ev,
                    'top_p1': top_p1,
                    'prob_gap': prob_gap,
                    'venue_code': vcode,
                    'cluster_id': c_id,
                    'wind_speed': w_speed,
                    'wave_height': w_height,
                    'b1_motor_rate': b1_motor,
                    'b1_boat_rate': b1_boat,
                    'b1_nat_win_rate': b1_nat_win,
                    'b1_local_win_rate': b1_local_win,
                    'b1_weight': b1_weight,
                    'b1_ex_momentum_diff': b1_mom_diff,
                    'b1_ex_diff_from_race_min': b1_diff_min,
                    'b1_ex_momentum_deviation': b1_mom_dev,
                    'wind_nige_vulnerability': w_nige_vuln,
                    'wind_makuri_cross': w_makuri_cr,
                    'high_wave_inner_risk': hw_risk,
                    'wave_weight_prod': wave_w_prod,
                    'is_hit': int(is_hit),
                    'payout': payout,
                    'profit': profit,
                    'ret_rate': ret_rate
                })

    df_bets = pd.DataFrame(bet_records)
    print(f"      -> 買い目候補抽出完了: {len(df_bets):,} 件のベット候補 ({time.time()-t0:.2f}秒)")
    
    # 全体ベースライン指標
    total_bets = len(df_bets)
    total_hits = df_bets['is_hit'].sum()
    hit_rate = total_hits / total_bets if total_bets > 0 else 0
    total_invest = total_bets * 100.0
    total_payout = df_bets['payout'].sum()
    total_profit = df_bets['profit'].sum()
    overall_roi = total_payout / total_invest if total_invest > 0 else 0

    print("\n" + "=" * 85)
    print(" 📊 【全体ベースライン（全ベット候補）成績】")
    print("=" * 85)
    print(f"  ・総ベット候補数 : {total_bets:,} 点 (均等100円投資)")
    print(f"  ・的中数 / 的中率: {total_hits:,} 点 ({hit_rate:.2%})")
    print(f"  ・総投資額       : {total_invest:,.0f} 円")
    print(f"  ・総払戻額       : {total_payout:,.0f} 円")
    print(f"  ・純損益         : {total_profit:+,.0f} 円")
    print(f"  ・通算回収率(ROI): {overall_roi:.2%}")
    print("=" * 85)

    # 5. 決定木（DecisionTreeRegressor）による条件抽出 (Sniper Approach)
    print("\n[5/5] 決定木（DecisionTreeRegressor: max_depth=3 & 4）を学習中...", flush=True)
    
    feature_cols = [
        'top_p1', 'prob_gap', 'ev', 'odds', 'prob', 'venue_code', 'cluster_id',
        'wind_speed', 'wave_height', 'b1_motor_rate', 'b1_boat_rate',
        'b1_nat_win_rate', 'b1_local_win_rate', 'b1_weight',
        'b1_ex_momentum_diff', 'b1_ex_diff_from_race_min', 'b1_ex_momentum_deviation',
        'wind_nige_vulnerability', 'wind_makuri_cross', 'high_wave_inner_risk', 'wave_weight_prod'
    ]

    X = df_bets[feature_cols]
    y = df_bets['ret_rate']  # 回収率（払戻/投資）を目的変数として学習

    # -------------------------------------------------------------
    # 決定木モデル学習 (深さ3)
    # -------------------------------------------------------------
    dt3 = DecisionTreeRegressor(max_depth=3, min_samples_leaf=30, random_state=42)
    dt3.fit(X, y)

    print("\n" + "=" * 90)
    print(" 🌲 【決定木 分岐ルール（Depth=3: Sniper Rules）】")
    print("=" * 90)
    tree_rules_3 = export_text(dt3, feature_names=feature_cols)
    print(tree_rules_3)

    # 各葉（Leaf）ノードごとの集計
    df_bets['leaf_id_3'] = dt3.apply(X)
    leaf_summary_3 = df_bets.groupby('leaf_id_3').agg(
        sample_count=('is_hit', 'count'),
        hits=('is_hit', 'sum'),
        total_bet=('payout', lambda x: len(x) * 100.0),
        total_payout=('payout', 'sum'),
        total_profit=('profit', 'sum'),
        mean_roi=('ret_rate', lambda x: x.mean() * 100.0)
    ).reset_index()

    leaf_summary_3['hit_rate'] = leaf_summary_3['hits'] / leaf_summary_3['sample_count'] * 100.0
    leaf_summary_3 = leaf_summary_3.sort_values('mean_roi', ascending=False)

    print("\n" + "=" * 110)
    print(" 🎯 【葉（Leaf）ノード別 パフォーマンス詳細一覧 (Depth=3)】")
    print("-" * 110)
    print(f"{'Leaf ID':<8} | {'サンプル数':<8} | {'的中数':<6} | {'的中率':<8} | {'総投資額(円)':<12} | {'総払戻額(円)':<12} | {'純損益(円)':<12} | {'期待回収率(ROI)':<12}")
    print("-" * 110)

    for idx, row in leaf_summary_3.iterrows():
        roi_badge = "🔥 [超高回収]" if row['mean_roi'] >= 115.0 else ("✨ [プラス]" if row['mean_roi'] >= 100.0 else "❌ [回収不足]")
        print(f"Node {int(row['leaf_id_3']):<3} | {int(row['sample_count']):<8,d} | {int(row['hits']):<6,d} | {row['hit_rate']:<7.2f}% | {row['total_bet']:<12,.0f} | {row['total_payout']:<12,.0f} | {row['total_profit']:<+12,.0f} | {row['mean_roi']:<6.2f}% {roi_badge}")

    print("-" * 110)

    # -------------------------------------------------------------
    # 決定木モデル学習 (深さ4 - より詳細なマイクロクラスタ)
    # -------------------------------------------------------------
    dt4 = DecisionTreeRegressor(max_depth=4, min_samples_leaf=25, random_state=42)
    dt4.fit(X, y)
    
    print("\n" + "=" * 90)
    print(" 🌲 【決定木 分岐ルール（Depth=4: Fine-grained Sniper Rules）】")
    print("=" * 90)
    tree_rules_4 = export_text(dt4, feature_names=feature_cols)
    print(tree_rules_4)

    df_bets['leaf_id_4'] = dt4.apply(X)
    leaf_summary_4 = df_bets.groupby('leaf_id_4').agg(
        sample_count=('is_hit', 'count'),
        hits=('is_hit', 'sum'),
        total_bet=('payout', lambda x: len(x) * 100.0),
        total_payout=('payout', 'sum'),
        total_profit=('profit', 'sum'),
        mean_roi=('ret_rate', lambda x: x.mean() * 100.0)
    ).reset_index()

    leaf_summary_4['hit_rate'] = leaf_summary_4['hits'] / leaf_summary_4['sample_count'] * 100.0
    leaf_summary_4 = leaf_summary_4.sort_values('mean_roi', ascending=False)

    print("\n" + "=" * 110)
    print(" 🎯 【葉（Leaf）ノード別 パフォーマンス詳細一覧 (Depth=4)】")
    print("-" * 110)
    print(f"{'Leaf ID':<8} | {'サンプル数':<8} | {'的中数':<6} | {'的中率':<8} | {'総投資額(円)':<12} | {'総払戻額(円)':<12} | {'純損益(円)':<12} | {'期待回収率(ROI)':<12}")
    print("-" * 110)

    for idx, row in leaf_summary_4.iterrows():
        roi_badge = "🔥 [超高回収]" if row['mean_roi'] >= 120.0 else ("✨ [プラス]" if row['mean_roi'] >= 100.0 else "❌ [回収不足]")
        print(f"Node {int(row['leaf_id_4']):<3} | {int(row['sample_count']):<8,d} | {int(row['hits']):<6,d} | {row['hit_rate']:<7.2f}% | {row['total_bet']:<12,.0f} | {row['total_payout']:<12,.0f} | {row['total_profit']:<+12,.0f} | {row['mean_roi']:<6.2f}% {roi_badge}")
    print("-" * 110)

    # 上位高回収率クラスタの条件ルールを詳しく解析・表示
    top_leaf_id = int(leaf_summary_4.iloc[0]['leaf_id_4'])
    top_leaf_roi = leaf_summary_4.iloc[0]['mean_roi']
    top_leaf_bets = leaf_summary_4.iloc[0]['sample_count']
    top_leaf_profit = leaf_summary_4.iloc[0]['total_profit']
    
    print("\n" + "=" * 90)
    print(f" 🎯 【最高回収率クラスタ (Node {top_leaf_id}) の詳細プロファイル】")
    print("=" * 90)
    print(f"  ・回収率 (ROI)    : {top_leaf_roi:.2f}%")
    print(f"  ・参戦点数        : {int(top_leaf_bets):,d} 点")
    print(f"  ・純利益          : {top_leaf_profit:+,.0f} 円")
    
    top_df = df_bets[df_bets['leaf_id_4'] == top_leaf_id]
    print(f"  ・平均EV          : {top_df['ev'].mean():.3f} (Min: {top_df['ev'].min():.2f}, Max: {top_df['ev'].max():.2f})")
    print(f"  ・平均オッズ      : {top_df['odds'].mean():.1f} 倍 (Min: {top_df['odds'].min():.1f}, Max: {top_df['odds'].max():.1f})")
    print(f"  ・平均風速        : {top_df['wind_speed'].mean():.2f} m")
    print(f"  ・平均波高        : {top_df['wave_height'].mean():.2f} cm")
    print(f"  ・1号艇モメンタム : {top_df['b1_ex_momentum_diff'].mean():.3f}s")
    print(f"  ・主な会場        : {dict(top_df['venue_code'].value_counts().head(5))}")
    print(f"  ・主な買い目構成  : {dict(top_df['combo'].value_counts().head(5))}")
    print("=" * 90)

    print(f"\n✅ 全工程完了 (総所要時間: {time.time()-t_start:.2f}秒)\n")

if __name__ == '__main__':
    run_micro_cluster_analysis()
