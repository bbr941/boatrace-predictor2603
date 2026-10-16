"""
backtest_gogo_lamp_conditions.py
的中特化（動的ダッチング / 累積50% / 最大8点 / 予算1,000円）において、
黄金ベースラインの条件（P1、難水面除外、EV、期待回収率等）を緩和した
「回収率100%超を達成するGOGOランプ点灯条件」を決定するためのバックテストスクリプト。
"""

import os
import sys
import time
import pickle
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import lightgbm as lgb

import train_model
from odds_normalizer import probs_to_init_scores
from probability_calibration import (
    calculate_benter_probs,
    get_default_calibrator,
    load_benter_cluster_config,
    get_cluster_benter_params
)
from portfolio_optimizer import calculate_dutching_bets

ODDS_CACHE_PATH = 'odds_cache_2026.pkl'
DATA_PATH = 'train_data_full.csv'
MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'
CLUSTER_1_VENUES = [2, 3, 4, 14, 22]  # 戸田, 江戸川, 平和島, 鳴門, 福岡


def run_gogo_lamp_backtest(sample_limit: int = 0):
    print("=" * 110)
    print(" 🚀 【的中特化】GOGOランプ点灯条件（回収率100%超 緩和化）バックテスト検証")
    print("=" * 110)
    t_start = time.time()

    # 1. 2026年テストデータ読み込み
    print("[1/5] 2026年データセット読み込み中...", end=" ", flush=True)
    t0 = time.time()
    df_all = pd.read_csv(DATA_PATH)
    df_test = df_all[df_all['race_date'] >= '2026-01-01'].copy()
    print(f"完了! ({len(df_test):,}行 / {df_test['race_id'].nunique():,}レース, {time.time()-t0:.2f}秒)")

    # 2. 確定着順 (1-3着)
    print("[2/5] 確定着順 & 各艇メタ情報を抽出中...", end=" ", flush=True)
    t0 = time.time()
    df_top3 = df_test[df_test['rank'].isin([1, 2, 3])].sort_values(['race_id', 'rank'])
    actual_combos = (
        df_top3.groupby('race_id')['boat_number']
        .apply(lambda s: '-'.join(map(str, s.astype(int))))
        .to_dict()
    )
    b1_dict = df_test[df_test['boat_number'] == 1].set_index('race_id').to_dict('index')
    print(f"完了! ({time.time()-t0:.2f}秒)")

    # 3. 直前オッズキャッシュ
    print("[3/5] オッズキャッシュのロード中...", end=" ", flush=True)
    t0 = time.time()
    with open(ODDS_CACHE_PATH, 'rb') as f:
        odds_cache = pickle.load(f)
    print(f"完了! ({len(odds_cache):,}レース, {time.time()-t0:.2f}秒)")

    # 4. モデル推論 (本命モデル P1 & 残差モデル Benter確率)
    print("[4/5] モデル推論 & Benter確率の一括準備中...", end=" ", flush=True)
    t0 = time.time()
    test_df = train_model.preprocess_data(df_test)
    if 'syn_win_rate' not in test_df.columns:
        test_df['syn_win_rate'] = 0.0
    race_sums = test_df.groupby('race_id')['syn_win_rate'].transform('sum')
    p_norm = np.where(race_sums > 0, test_df['syn_win_rate'] / np.maximum(race_sums, 1e-9), 1.0 / 6.0)
    test_df['init_score'] = probs_to_init_scores(p_norm, clip_eps=1e-5)

    model_h = lgb.Booster(model_file=MODEL_HONMEI_PATH)
    model_r = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)

    test_df['score_h'] = model_h.predict(test_df[model_h.feature_name()])
    test_df['raw_res'] = model_r.predict(test_df[model_r.feature_name()], raw_score=True)
    test_df['total_logits'] = test_df['raw_res'] + test_df['init_score'].to_numpy()
    p_raw = 1.0 / (1.0 + np.exp(-np.clip(test_df['total_logits'].to_numpy(), -30, 30)))
    test_df['p_raw'] = p_raw
    race_p_sums = test_df.groupby('race_id')['p_raw'].transform('sum')
    test_df['p_norm_res'] = test_df['p_raw'] / np.maximum(race_p_sums, 1e-9)

    calibrator = get_default_calibrator('platt')
    cluster_cfg = load_benter_cluster_config()
    cluster_lookup = {v: get_cluster_benter_params(v, cluster_cfg) for v in range(1, 25)}
    print(f"完了! ({time.time()-t0:.2f}秒)")

    # 5. 的中特化買い目生成 & 評価特徴量の算出
    print("[5/5] 全レースの的中特化買い目生成 & 評価中...", end=" ", flush=True)
    t0 = time.time()

    all_race_results = []
    race_groups = list(test_df.groupby('race_id', sort=False))
    if sample_limit > 0:
        race_groups = race_groups[:sample_limit]

    count = 0
    for rid, grp in race_groups:
        if rid not in odds_cache or rid not in actual_combos or rid not in b1_dict:
            continue

        b1_info = b1_dict[rid]
        v_code = int(b1_info.get('venue_code', 1))
        r_date = str(b1_info.get('race_date', ''))
        b1_motor = float(b1_info.get('motor_rate', 0.0))
        wave_h = float(b1_info.get('wave_height', 0.0))

        # 本命モデル P1 (キャリブレーション後)
        s_dict = dict(zip(grp['boat_number'], grp['score_h']))
        p_h = calibrator.calibrate_scores(s_dict)
        p1_val = p_h.get(1, 0.0)  # 1号艇の勝率
        sorted_p = sorted(p_h.values(), reverse=True)
        max_p = sorted_p[0] if sorted_p else 0.0
        second_p = sorted_p[1] if len(sorted_p) > 1 else 0.0
        prob_gap = max_p - second_p

        # Benter確率の展開
        odds = odds_cache[rid]
        actual_combo = actual_combos[rid]
        d2_c, d3_c, c_id, c_name = cluster_lookup.get(v_code, (0.40, 0.60, 2, '標準水面'))
        p1_dict_res = dict(zip(grp['boat_number'], grp['p_norm_res']))
        benter_probs, _, _ = calculate_benter_probs(p1_dict_res, d2=d2_c, d3=d3_c, calibration_method='direct')
        benter_probs_dict = {p['combo']: p['prob'] for p in benter_probs}

        # 的中特化買い目の生成 (完全的中特化: min_synthetic_odds=None)
        bets = calculate_dutching_bets(
            benter_probs=benter_probs_dict,
            odds_dict=odds,
            budget=1000,
            target_cum_prob=0.50,
            max_combos=8,
            min_combos=2,
            min_synthetic_odds=None
        )

        if not bets:
            continue

        total_bet = sum(bets.values())
        is_hit = (actual_combo in bets)
        actual_odds = odds.get(actual_combo, 0.0) if is_hit else 0.0
        payout = int((bets[actual_combo] / 100.0) * actual_odds * 100.0) if is_hit else 0
        profit = payout - total_bet

        # 特徴量計算
        # 1. 買い目の合成期待回収率 (Expected ROI)
        expected_payout = sum(
            benter_probs_dict.get(c, 0.0) * odds.get(c, 0.0) * amt
            for c, amt in bets.items()
        )
        expected_roi = expected_payout / total_bet if total_bet > 0 else 0.0

        # 2. 買い目内の最大EV
        ev_list = [benter_probs_dict.get(c, 0.0) * odds.get(c, 0.0) for c in bets.keys()]
        max_ev = max(ev_list) if ev_list else 0.0
        avg_ev = np.mean(ev_list) if ev_list else 0.0
        # EV >= 1.0 の買い目数
        count_ev_ge_1 = sum(1 for ev in ev_list if ev >= 1.0)
        has_positive_ev = (count_ev_ge_1 > 0)

        # 3. 最低オッズ・最高オッズ
        selected_odds = [odds.get(c, 0.0) for c in bets.keys()]
        min_odds = min(selected_odds) if selected_odds else 0.0
        max_odds = max(selected_odds) if selected_odds else 0.0

        # 4. 難水面フラグ
        is_cluster_1 = (v_code in CLUSTER_1_VENUES)

        all_race_results.append({
            'race_id': rid,
            'race_date': r_date,
            'venue_code': v_code,
            'b1_motor': b1_motor,
            'wave_h': wave_h,
            'p1': p1_val,
            'max_p': max_p,
            'prob_gap': prob_gap,
            'is_cluster_1': is_cluster_1,
            'total_bet': total_bet,
            'bets_count': len(bets),
            'is_hit': is_hit,
            'payout': payout,
            'profit': profit,
            'expected_roi': expected_roi,
            'max_ev': max_ev,
            'avg_ev': avg_ev,
            'has_positive_ev': has_positive_ev,
            'count_ev_ge_1': count_ev_ge_1,
            'min_odds': min_odds,
            'max_odds': max_odds
        })

        count += 1
        if count % 5000 == 0:
            print(f"{count:,}レース...", end=" ", flush=True)

    print(f"完了! (評価レース数: {len(all_race_results):,}, 処理時間: {time.time()-t0:.2f}秒)\n")

    df_res = pd.DataFrame(all_race_results)

    # 6. 様々な条件での集計
    def evaluate_condition(name: str, mask: pd.Series):
        sub = df_res[mask]
        n_races = len(sub)
        if n_races == 0:
            return {
                '条件名': name, 'レース数': 0, '選定率(%)': 0.0,
                '的中数': 0, '的中率(%)': 0.0, '総投資額': 0,
                '総払戻額': 0, '純損益': 0, '回収率(%)': 0.0, 'PF': 0.0
            }
        hits = sub['is_hit'].sum()
        hit_rate = (hits / n_races) * 100.0
        tot_bet = sub['total_bet'].sum()
        tot_pay = sub['payout'].sum()
        net_prof = tot_pay - tot_bet
        roi = (tot_pay / tot_bet) * 100.0 if tot_bet > 0 else 0.0
        select_rate = (n_races / len(df_res)) * 100.0
        
        # Profit Factor
        gross_profit = sub[sub['profit'] > 0]['profit'].sum()
        gross_loss = abs(sub[sub['profit'] < 0]['profit'].sum())
        pf = (gross_profit / gross_loss) if gross_loss > 0 else 99.9

        return {
            '条件名': name,
            'レース数': n_races,
            '選定率(%)': round(select_rate, 1),
            '的中数': hits,
            '的中率(%)': round(hit_rate, 2),
            '総投資額': tot_bet,
            '総払戻額': tot_pay,
            '純損益': net_prof,
            '回収率(%)': round(roi, 2),
            'PF': round(pf, 3)
        }

    # テストする条件一覧
    conditions = [
        ("【ベースライン】全レース的中特化 (無条件)", pd.Series(True, index=df_res.index)),
        
        # 1. 期待回収率 (Expected ROI) 単体
        ("ExpROI >= 1.00 (期待値プラス)", df_res['expected_roi'] >= 1.00),
        ("ExpROI >= 1.05", df_res['expected_roi'] >= 1.05),
        ("ExpROI >= 1.10", df_res['expected_roi'] >= 1.10),
        ("ExpROI >= 1.15", df_res['expected_roi'] >= 1.15),
        ("ExpROI >= 1.20", df_res['expected_roi'] >= 1.20),

        # 2. Gatekeeper P1 緩和単体
        ("P1 >= 0.50", df_res['p1'] >= 0.50),
        ("P1 >= 0.55", df_res['p1'] >= 0.55),
        ("P1 >= 0.60", df_res['p1'] >= 0.60),
        ("P1 >= 0.65", df_res['p1'] >= 0.65),
        ("P1 >= 0.70", df_res['p1'] >= 0.70),
        ("P1 >= 0.7438 (黄金GK水準)", df_res['p1'] >= 0.7438),

        # 3. 難水面除外単体
        ("難水面除外 (非Cluster1)", ~df_res['is_cluster_1']),

        # 4. 最大EV単体
        ("Max EV >= 1.00 (プラスEV買い目あり)", df_res['max_ev'] >= 1.00),
        ("Max EV >= 1.10", df_res['max_ev'] >= 1.10),
        ("Max EV >= 1.20", df_res['max_ev'] >= 1.20),

        # 5. 複合条件（本命信頼度 × 期待値）
        ("P1 >= 0.50 & ExpROI >= 1.00", (df_res['p1'] >= 0.50) & (df_res['expected_roi'] >= 1.00)),
        ("P1 >= 0.55 & ExpROI >= 1.00", (df_res['p1'] >= 0.55) & (df_res['expected_roi'] >= 1.00)),
        ("P1 >= 0.60 & ExpROI >= 1.00", (df_res['p1'] >= 0.60) & (df_res['expected_roi'] >= 1.00)),
        ("P1 >= 0.65 & ExpROI >= 1.00", (df_res['p1'] >= 0.65) & (df_res['expected_roi'] >= 1.00)),
        ("P1 >= 0.70 & ExpROI >= 1.00", (df_res['p1'] >= 0.70) & (df_res['expected_roi'] >= 1.00)),

        # 6. 複合条件（本命信頼度 × 難水面除外 × ExpROI）
        ("非難水面 & P1 >= 0.55 & ExpROI >= 1.00", (~df_res['is_cluster_1']) & (df_res['p1'] >= 0.55) & (df_res['expected_roi'] >= 1.00)),
        ("非難水面 & P1 >= 0.60 & ExpROI >= 1.00", (~df_res['is_cluster_1']) & (df_res['p1'] >= 0.60) & (df_res['expected_roi'] >= 1.00)),
        ("非難水面 & P1 >= 0.60 & ExpROI >= 1.05", (~df_res['is_cluster_1']) & (df_res['p1'] >= 0.60) & (df_res['expected_roi'] >= 1.05)),
        ("非難水面 & P1 >= 0.60 & ExpROI >= 1.10", (~df_res['is_cluster_1']) & (df_res['p1'] >= 0.60) & (df_res['expected_roi'] >= 1.10)),

        # 7. 複合条件（P1 × Max EV）
        ("P1 >= 0.55 & Max EV >= 1.10", (df_res['p1'] >= 0.55) & (df_res['max_ev'] >= 1.10)),
        ("P1 >= 0.60 & Max EV >= 1.10", (df_res['p1'] >= 0.60) & (df_res['max_ev'] >= 1.10)),
        ("非難水面 & P1 >= 0.60 & Max EV >= 1.10", (~df_res['is_cluster_1']) & (df_res['p1'] >= 0.60) & (df_res['max_ev'] >= 1.10)),
        ("非難水面 & P1 >= 0.60 & Max EV >= 1.15", (~df_res['is_cluster_1']) & (df_res['p1'] >= 0.60) & (df_res['max_ev'] >= 1.15)),

        # 8. 黄金ベースライン完全条件 (参考)
        ("【参考: 黄金GK】P1 >= 0.7438 & 非難水面", (df_res['p1'] >= 0.7438) & (~df_res['is_cluster_1'])),
    ]

    records = [evaluate_condition(name, mask) for name, mask in conditions]
    df_eval = pd.DataFrame(records)

    print("\n" + "=" * 110)
    print(" 📊 【バックテスト集計結果】")
    print("=" * 110)
    fmt_cols = ['条件名', 'レース数', '選定率(%)', '的中率(%)', '総投資額', '総払戻額', '純損益', '回収率(%)', 'PF']
    print(df_eval[fmt_cols].to_string(index=False))

    # 回収率100%超の条件をピックアップ
    over_100 = df_eval[df_eval['回収率(%)'] >= 100.0].sort_values(by='回収率(%)', ascending=False)
    print("\n" + "=" * 110)
    print(f" 🏆 【回収率100%超を達成した条件 (全{len(over_100)}件)】")
    print("=" * 110)
    if len(over_100) > 0:
        print(over_100[fmt_cols].to_string(index=False))
    else:
        print("※ 単純条件では100%超が見つかりませんでした。より詳細な条件探索を推奨します。")

    # CSV保存
    df_eval.to_csv("backtest_gogo_lamp_results.csv", index=False, encoding="utf-8-sig")
    print("\n💾 詳細結果を backtest_gogo_lamp_results.csv に保存しました。")
    print(f"⏱️ 総所要時間: {time.time()-t_start:.2f}秒")


if __name__ == '__main__':
    run_gogo_lamp_backtest()
