"""
optimize_probability.py
Benterモデル (d2, d3) および確率キャリブレーションの最適パラメーター探索スクリプト (Optuna)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sqlite3
import argparse
import json
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna

import train_model
from probability_calibration import (
    BoatRaceCalibrator,
    calculate_benter_probs,
    save_probability_config,
    CONFIG_PATH,
    CALIBRATOR_MODEL_PATH
)
from simulate_betting import (
    select_hybrid_formation,
    calculate_funds_distribution
)

# ログ詳細度設定
optuna.logging.set_verbosity(optuna.logging.WARNING)

MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_ANA_PATH = 'model_ana.txt'
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DB_PATH = 'boatrace.db'


def get_db_connection():
    if os.path.exists(DB_PATH):
        return sqlite3.connect(DB_PATH)
    alt_path = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
    if os.path.exists(alt_path):
        return sqlite3.connect(alt_path)
    return sqlite3.connect(DB_PATH)


def load_and_prepare_data(max_races=2000):
    """
    データセットおよびDBからレース結果・オッズを読み込み、推論スコアを事前計算してキャッシュする
    """
    print(f"=== 1. データ読み込み & スコア事前計算 (上限: {max_races} レース) ===", flush=True)
    if not os.path.exists(MODEL_HONMEI_PATH) or not os.path.exists(MODEL_ANA_PATH):
        raise FileNotFoundError("モデルファイル (model_honmei.txt / model_ana.txt) が見つかりません。")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT race_id FROM odds_data ORDER BY race_id DESC LIMIT ?", [max_races])
    valid_races = [row[0] for row in cursor.fetchall()]
    valid_races_set = set(valid_races)
    print(f"DB内の対象レース数: {len(valid_races)} レース", flush=True)

    print("CSVデータセットから対象レースを高速抽出中...", flush=True)
    chunks = []
    for chunk in pd.read_csv(DATA_PATH, chunksize=150000):
        matched = chunk[chunk['race_id'].isin(valid_races_set)]
        if len(matched) > 0:
            chunks.append(matched)

    if not chunks:
        raise ValueError("対象レースがCSVデータセット内に見つかりませんでした。")

    df_matched = pd.concat(chunks, ignore_index=True)
    print(f"抽出完了: {len(df_matched)} 行 ({df_matched['race_id'].nunique()} レース)", flush=True)

    print("特徴量前処理を実行中...", flush=True)
    test_df = train_model.preprocess_data(df_matched)
    test_races = test_df['race_id'].unique()
    print(f"前処理完了: {len(test_races)} レース", flush=True)

    # LightGBM推論
    print("本命モデル & 穴モデルによるスコア推論実行中...", flush=True)
    model_honmei = lgb.Booster(model_file=MODEL_HONMEI_PATH)
    model_ana = lgb.Booster(model_file=MODEL_ANA_PATH)

    feats_h = model_honmei.feature_name()
    for f in feats_h:
        if f not in test_df.columns:
            test_df[f] = 0
    test_df['score_honmei'] = model_honmei.predict(test_df[feats_h])

    feats_a = model_ana.feature_name()
    for f in feats_a:
        if f not in test_df.columns:
            test_df[f] = 0
    test_df['score_ana'] = model_ana.predict(test_df[feats_a])

    # 1着フラグ (キャリブレーター学習用)
    test_df['is_1st'] = (test_df['rank'] == 1).astype(int)

    # レース毎のデータキャッシュ作成
    print("オッズおよびレース確定着順をキャッシュ中...", flush=True)
    race_cache = []
    groups = test_df.groupby('race_id')

    for rid, group in groups:
        honmei_scores = dict(zip(group['boat_number'], group['score_honmei']))
        ana_scores = dict(zip(group['boat_number'], group['score_ana']))

        # 確定3連単着順の取得
        try:
            r1 = group[group['rank'] == 1]['boat_number'].iloc[0]
            r2 = group[group['rank'] == 2]['boat_number'].iloc[0]
            r3 = group[group['rank'] == 3]['boat_number'].iloc[0]
            actual_combo = f"{int(r1)}-{int(r2)}-{int(r3)}"
            actual_1st = int(r1)
        except (IndexError, KeyError):
            actual_combo = None
            actual_1st = None

        # オッズ取得 (DB)
        cursor.execute("SELECT combination, odds_1min FROM odds_data WHERE race_id = ? AND length(combination) = 3", [rid])
        rows = cursor.fetchall()
        odds_map = {}
        for r in rows:
            comb_db = str(r[0])
            if len(comb_db) == 3:
                odds_map[f"{comb_db[0]}-{comb_db[1]}-{comb_db[2]}"] = float(r[1])

        race_cache.append({
            'race_id': rid,
            'honmei_scores': honmei_scores,
            'ana_scores': ana_scores,
            'actual_combo': actual_combo,
            'actual_1st': actual_1st,
            'all_odds': odds_map,
            'group_df': group[['boat_number', 'score_honmei', 'is_1st']].copy()
        })

    conn.close()
    print(f"キャッシュ構築完了: {len(race_cache)} レース", flush=True)
    return race_cache, test_df


def train_calibrators(test_df):
    """
    過去スコアデータからPlatt ScalingおよびIsotonic Regressionキャリブレーターを学習・保存する
    """
    print("\n=== 2. 確率キャリブレーターの学習 & 保存 ===", flush=True)
    X_scores = test_df['score_honmei'].to_numpy()
    y_true = test_df['is_1st'].to_numpy()

    calibrator_platt = BoatRaceCalibrator(method='platt').fit(X_scores, y_true)
    calibrator_iso = BoatRaceCalibrator(method='isotonic').fit(X_scores, y_true)

    # デフォルトとしてPlatt Scalingモデルを保存
    calibrator_platt.save(CALIBRATOR_MODEL_PATH)
    print(f"キャリブレーションモデルを保存しました: {CALIBRATOR_MODEL_PATH}", flush=True)

    calibrators = {
        'platt': calibrator_platt,
        'isotonic': calibrator_iso,
        'softmax': BoatRaceCalibrator(method='softmax')
    }
    return calibrators


def evaluate_simulation(race_cache, d2, d3, calibrator, use_plan_b=True, p1_th=0.49, gap_th=0.010):
    """
    与えられた (d2, d3, calibrator) に基づいて全キャッシュレースのバックテストを行い、
    ROI, 回収額, 的中率, 3連単LogLossを算出する。
    """
    stats = {
        'total': len(race_cache),
        'betted': 0,
        'hits': 0,
        'bet_amt': 0,
        'return_amt': 0,
        'log_loss_sum': 0.0,
        'log_loss_count': 0
    }

    for r in race_cache:
        probs, max_p1, prob_gap = calculate_benter_probs(
            r['honmei_scores'],
            d2=d2,
            d3=d3,
            calibrator=calibrator
        )

        # Log Loss 計算 (全レース対象)
        if r['actual_combo']:
            actual_p = next((p['prob'] for p in probs if p['combo'] == r['actual_combo']), 1e-6)
            stats['log_loss_sum'] += -math.log(max(actual_p, 1e-6))
            stats['log_loss_count'] += 1

        # ベッティング判定 (Plan B フィルター)
        if use_plan_b:
            if max_p1 < p1_th or prob_gap < gap_th:
                continue

        all_odds = r['all_odds']
        if not all_odds:
            continue

        selected_combos = select_hybrid_formation(probs, r['ana_scores'], all_odds)
        if not selected_combos:
            continue

        bets = calculate_funds_distribution(selected_combos, probs, all_odds)
        if not bets:
            continue

        stats['betted'] += 1
        stats['bet_amt'] += sum(bets.values())

        if r['actual_combo'] and r['actual_combo'] in bets:
            stats['hits'] += 1
            stats['return_amt'] += bets[r['actual_combo']] * all_odds.get(r['actual_combo'], 0)

    roi = (stats['return_amt'] / stats['bet_amt']) if stats['bet_amt'] > 0 else 0.0
    hit_rate = (stats['hits'] / stats['betted']) if stats['betted'] > 0 else 0.0
    avg_log_loss = (stats['log_loss_sum'] / stats['log_loss_count']) if stats['log_loss_count'] > 0 else 999.0
    profit = stats['return_amt'] - stats['bet_amt']

    return {
        'roi': roi,
        'profit': profit,
        'hit_rate': hit_rate,
        'betted': stats['betted'],
        'log_loss': avg_log_loss,
        'stats': stats
    }


def run_optuna_optimization(race_cache, calibrators, n_trials=50, metric='roi', use_plan_b=True):
    """
    Optunaによるハイパーパラメータ探索
    """
    print(f"\n=== 3. Optuna 最適パラメータ探索開始 (Trials: {n_trials}, 目的関数: {metric.upper()}) ===", flush=True)

    def objective(trial):
        d2 = trial.suggest_float('d2', 0.2, 1.5, step=0.05)
        d3 = trial.suggest_float('d3', 0.2, 1.5, step=0.05)
        calib_method = trial.suggest_categorical('calibration_method', ['platt', 'isotonic', 'softmax'])
        
        calibrator = calibrators[calib_method]
        res = evaluate_simulation(
            race_cache,
            d2=d2,
            d3=d3,
            calibrator=calibrator,
            use_plan_b=use_plan_b
        )

        if metric == 'roi':
            if res['betted'] < 10:
                return 0.0
            return res['roi']
        elif metric == 'profit':
            return res['profit']
        elif metric == 'log_loss':
            return res['log_loss']
        else:
            return res['roi']

    direction = 'minimize' if metric == 'log_loss' else 'maximize'
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)

    print("\n=== 探索完了 ===", flush=True)
    print(f"Best Trial ({metric.upper()}): {study.best_value:.4f}", flush=True)
    print("Best Parameters:", flush=True)
    for k, v in study.best_params.items():
        print(f"  {k}: {v}", flush=True)

    return study.best_params


def run_optimization_pipeline(max_races=2000, n_trials=50, metric='roi', use_plan_b=True, save_config=True):
    race_cache, test_df = load_and_prepare_data(max_races=max_races)
    calibrators = train_calibrators(test_df)

    # 1. ベースライン (従来のPlackett-Luce: d2=1.0, d3=1.0, Softmax未補正)
    print("\n--- [基準値] 従来のPlackett-Luce (d2=1.0, d3=1.0, Softmax) ---", flush=True)
    base_res = evaluate_simulation(
        race_cache,
        d2=1.0,
        d3=1.0,
        calibrator=calibrators['softmax'],
        use_plan_b=use_plan_b
    )
    print(f"Betted Races: {base_res['betted']}/{len(race_cache)}", flush=True)
    print(f"Hit Rate: {base_res['hit_rate']:.2%}", flush=True)
    print(f"ROI: {base_res['roi']:.2%}", flush=True)
    print(f"Total Profit: {int(base_res['profit']):,} JPY", flush=True)
    print(f"3-Ren-Tan Log Loss: {base_res['log_loss']:.4f}", flush=True)

    # 2. Optuna最適化
    best_params = run_optuna_optimization(
        race_cache,
        calibrators,
        n_trials=n_trials,
        metric=metric,
        use_plan_b=use_plan_b
    )

    # 3. 最適化モデルの評価
    best_calibrator = calibrators[best_params['calibration_method']]
    opt_res = evaluate_simulation(
        race_cache,
        d2=best_params['d2'],
        d3=best_params['d3'],
        calibrator=best_calibrator,
        use_plan_b=use_plan_b
    )

    print(f"\n--- [最適化後] Benterモデル + キャリブレーション ({best_params['calibration_method']}) ---", flush=True)
    print(f"Parameters: d2={best_params['d2']}, d3={best_params['d3']}, method={best_params['calibration_method']}", flush=True)
    print(f"Betted Races: {opt_res['betted']}/{len(race_cache)}", flush=True)
    print(f"Hit Rate: {opt_res['hit_rate']:.2%}", flush=True)
    print(f"ROI: {opt_res['roi']:.2%}", flush=True)
    print(f"Total Profit: {int(opt_res['profit']):,} JPY", flush=True)
    print(f"3-Ren-Tan Log Loss: {opt_res['log_loss']:.4f}", flush=True)

    roi_diff = opt_res['roi'] - base_res['roi']
    profit_diff = opt_res['profit'] - base_res['profit']
    logloss_diff = opt_res['log_loss'] - base_res['log_loss']

    print(f"\n=== 改善度サマリー ===", flush=True)
    print(f"ROI変化: {base_res['roi']:.2%} -> {opt_res['roi']:.2%} ({roi_diff:+.2%})", flush=True)
    print(f"損益変化: {int(base_res['profit']):,} JPY -> {int(opt_res['profit']):,} JPY ({int(profit_diff):+,d} JPY)", flush=True)
    print(f"3連単Log Loss変化: {base_res['log_loss']:.4f} -> {opt_res['log_loss']:.4f} ({logloss_diff:+.4f})", flush=True)

    # 4. 設定ファイルの保存
    if save_config:
        config_to_save = {
            'calibration_method': best_params['calibration_method'],
            'd2': round(best_params['d2'], 3),
            'd3': round(best_params['d3'], 3),
            'p1_th': 0.49,
            'gap_th': 0.010,
            'optimized_metric': metric,
            'roi': round(opt_res['roi'], 4),
            'hit_rate': round(opt_res['hit_rate'], 4)
        }
        save_probability_config(config_to_save)
        print(f"\n最適設定を保存しました: {CONFIG_PATH}", flush=True)

    return {
        'base': base_res,
        'optimized': opt_res,
        'best_params': best_params
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize Benter Damping Factors and Probability Calibration using Optuna")
    parser.add_argument('--races', type=int, default=2000, help="Number of races to evaluate (default: 2000)")
    parser.add_argument('--n_trials', type=int, default=50, help="Number of Optuna trials (default: 50)")
    parser.add_argument('--metric', type=str, default='roi', choices=['roi', 'log_loss', 'profit'], help="Optimization metric")
    parser.add_argument('--plan_a', action='store_true', help="Use Plan A (all races) instead of Plan B")
    parser.add_argument('--no_save', action='store_true', help="Do not save config to file")
    
    args = parser.parse_args()
    run_optimization_pipeline(
        max_races=args.races,
        n_trials=args.n_trials,
        metric=args.metric,
        use_plan_b=not args.plan_a,
        save_config=not args.no_save
    )
