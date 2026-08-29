"""
experiment_features_sqlite.py
【実験用スクリプト】ローカルSQLite (boatrace.db) を用いた環境クロス特徴量の検証
- 本番稼働コードには一切手を加えない独立スクリプト
- 風速クロス (Wind Speed Cross)、波高クロス (Wave Height Cross)、代替モメンタム (Exhibition Momentum) を算出
- LightGBM によるベースライン vs 新規特徴量追加版の予測精度・Feature Importance 比較
"""

import os
import time
import sqlite3
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss

# データベースパス
LOCAL_DB_CANDIDATES = [
    r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db',
    'boatrace.db'
]

def get_sqlite_path() -> str:
    for path in LOCAL_DB_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("ローカル SQLite データベース (boatrace.db) が見つかりません。")


def extract_raw_data(db_path: str, start_date: str = '2024-01-01', limit_races: int = 60000) -> pd.DataFrame:
    """
    SQLiteから races, race_entries, results, before_info, Racer_CourseWinTech を高速結合抽出
    """
    print(f"  [1/4] SQLite ({db_path}) から生データを抽出中 (開始日: {start_date}, 上限: {limit_races:,} レース)...", flush=True)
    t0 = time.time()
    
    # 読み取り専用モードで接続 (ロック回避)
    norm_path = os.path.abspath(db_path).replace('\\', '/')
    try:
        con = sqlite3.connect(f"file:///{norm_path}?mode=ro", uri=True)
    except Exception:
        con = sqlite3.connect(db_path)


    query = f"""
    SELECT 
        r.race_id,
        r.venue_code,
        r.race_date,
        r.race_number,
        r.wind_direction,
        r.wind_speed,
        r.wave_height,
        r.weather,
        re.boat_number,
        re.racer_id,
        re.racer_name,
        re.racer_rank,
        re.age,
        re.weight,
        re.nat_win_rate,
        re.nat_quinella_rate,
        re.loc_win_rate,
        re.loc_quinella_rate,
        re.motor_rate,
        re.boat_rate,
        res.finish_order,
        bi.exhibition_time,
        bi.exhibition_start_timing,
        bi.exhibition_entry_course,
        rcw.RacesRun as rcw_races_run,
        rcw.Wins as rcw_wins,
        rcw.Nige as rcw_nige,
        rcw.Sashi as rcw_sashi,
        rcw.Makuri as rcw_makuri,
        rcw.Makurizashi as rcw_makurizashi
    FROM races r
    JOIN race_entries re ON r.race_id = re.race_id
    LEFT JOIN results res ON re.race_id = res.race_id AND re.boat_number = res.boat_number
    LEFT JOIN before_info bi ON re.race_id = bi.race_id AND re.boat_number = bi.boat_number
    LEFT JOIN Racer_CourseWinTech rcw ON re.racer_id = rcw.RacerID 
        AND COALESCE(bi.exhibition_entry_course, re.boat_number) = rcw.Course
    WHERE r.race_date >= '{start_date}' AND r.is_cancelled = 0
    ORDER BY r.race_date, r.race_number, re.boat_number
    LIMIT {limit_races * 6};
    """
    
    df = pd.read_sql_query(query, con)
    con.close()
    print(f"        -> 抽出行数: {len(df):,} 行 ({len(df)//6:,} レース相当 / 所要時間: {time.time()-t0:.2f}秒)", flush=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    環境クロス特徴量 & 代替モメンタム特徴量を生成
    """
    print("  [2/4] 特徴量エンジニアリング（環境クロス・代替モメンタム生成）中...", flush=True)
    t0 = time.time()
    
    # 欠損値補正 & 基本変換
    df['venue_code'] = pd.to_numeric(df['venue_code'], errors='coerce').fillna(1).astype(int)
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce').fillna(0.0)
    df['wave_height'] = pd.to_numeric(df['wave_height'], errors='coerce').fillna(0.0)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(52.0)
    df['nat_win_rate'] = pd.to_numeric(df['nat_win_rate'], errors='coerce').fillna(0.0)
    df['loc_win_rate'] = pd.to_numeric(df['loc_win_rate'], errors='coerce').fillna(0.0)
    df['motor_rate'] = pd.to_numeric(df['motor_rate'], errors='coerce').fillna(30.0)
    df['boat_rate'] = pd.to_numeric(df['boat_rate'], errors='coerce').fillna(30.0)
    
    rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
    df['racer_rank_num'] = df['racer_rank'].map(rank_map).fillna(2).astype(int)
    
    df['exhibition_time'] = pd.to_numeric(df['exhibition_time'], errors='coerce')
    # 展示タイムの欠損値は会場×艇番の中央値で補完
    df['exhibition_time'] = df.groupby(['venue_code', 'boat_number'])['exhibition_time'].transform(lambda x: x.fillna(x.median())).fillna(6.80)
    
    df['rcw_races_run'] = pd.to_numeric(df['rcw_races_run'], errors='coerce').fillna(0)
    df['rcw_wins'] = pd.to_numeric(df['rcw_wins'], errors='coerce').fillna(0)
    df['rcw_nige'] = pd.to_numeric(df['rcw_nige'], errors='coerce').fillna(0)
    df['rcw_sashi'] = pd.to_numeric(df['rcw_sashi'], errors='coerce').fillna(0)
    df['rcw_makuri'] = pd.to_numeric(df['rcw_makuri'], errors='coerce').fillna(0)
    df['rcw_makurizashi'] = pd.to_numeric(df['rcw_makurizashi'], errors='coerce').fillna(0)

    # 決まり手出現率 (該当コースでの実績比率)
    denom = np.maximum(df['rcw_races_run'], 1.0)
    df['makuri_rate'] = df['rcw_makuri'] / denom
    df['makurizashi_rate'] = df['rcw_makurizashi'] / denom
    df['sashi_rate'] = df['rcw_sashi'] / denom
    df['nige_rate'] = df['rcw_nige'] / denom

    # ----------------------------------------------------
    # ① 風速クロス (Wind Speed Cross)
    # ----------------------------------------------------
    # 強風フラグ (風速4m以上、6m以上)
    df['is_strong_wind'] = (df['wind_speed'] >= 4.0).astype(float)
    df['is_gale_wind'] = (df['wind_speed'] >= 6.0).astype(float)
    
    # 風速 × まくり実績
    df['wind_makuri_cross'] = df['wind_speed'] * df['makuri_rate']
    df['strong_wind_makuri'] = df['is_strong_wind'] * df['makuri_rate']
    
    # 風速 × まくり差し実績
    df['wind_makurizashi_cross'] = df['wind_speed'] * df['makurizashi_rate']
    
    # 強風 × 外枠（3〜6号艇）アドバンテージ
    df['strong_wind_outer_adv'] = df['is_strong_wind'] * (df['boat_number'] >= 3).astype(float)
    
    # 風速 × イン逃げ減衰リスク (風が強いのに逃げ率が低い場合のペナルティ)
    df['wind_nige_vulnerability'] = df['wind_speed'] * (1.0 - df['nige_rate']) * (df['boat_number'] == 1).astype(float)

    # ----------------------------------------------------
    # ② 波高クロス (Wave Height Cross)
    # ----------------------------------------------------
    # 波高 × 体重積 (波が高いときの重量負荷)
    df['wave_weight_prod'] = df['wave_height'] * df['weight']
    df['wave_weight_ratio'] = df['wave_height'] / np.maximum(df['weight'], 40.0)
    
    # 高波フラグ (波高4cm以上)
    df['is_high_wave'] = (df['wave_height'] >= 4.0).astype(float)
    # 高波 × 重量級ペナルティ (52kg基準の超過分)
    df['high_wave_heavy_penalty'] = df['is_high_wave'] * np.maximum(0.0, df['weight'] - 52.0)
    # 高波 × 1号艇旋回リスク
    df['high_wave_inner_risk'] = df['is_high_wave'] * (df['boat_number'] == 1).astype(float)

    # ----------------------------------------------------
    # ③ 代替モメンタム (Exhibition Momentum)
    # ----------------------------------------------------
    # レース内相対展示タイム
    race_min_ex = df.groupby('race_id')['exhibition_time'].transform('min')
    race_mean_ex = df.groupby('race_id')['exhibition_time'].transform('mean')
    df['ex_diff_from_race_min'] = df['exhibition_time'] - race_min_ex
    df['ex_diff_from_race_mean'] = df['exhibition_time'] - race_mean_ex
    df['ex_rank_in_race'] = df.groupby('race_id')['exhibition_time'].rank(method='min')

    # 節間（同一会場・同一選手）での展示タイムモメンタム
    # 日付・レース番号順に並んでいる状態で shift
    df['prev_ex_in_series'] = df.groupby(['venue_code', 'racer_id'])['exhibition_time'].shift(1)
    # 前走タイムとの差分 (マイナスならタイム短縮＝機力上向き)
    df['ex_momentum_diff'] = (df['exhibition_time'] - df['prev_ex_in_series']).fillna(0.0)
    
    # 節間累積平均展示タイムからの乖離
    series_exp_mean = df.groupby(['venue_code', 'racer_id'])['exhibition_time'].transform(lambda x: x.expanding().mean())
    df['ex_momentum_deviation'] = df['exhibition_time'] - series_exp_mean

    # 正解ラベル: 1着かどうか
    df['finish_order'] = pd.to_numeric(df['finish_order'], errors='coerce')
    df['is_win'] = (df['finish_order'] == 1).astype(int)

    print(f"        -> 特徴量生成完了 (所要時間: {time.time()-t0:.2f}秒)", flush=True)
    return df


def train_and_compare_models(df: pd.DataFrame, split_date: str = '2026-01-01'):
    """
    LightGBM でベースライン特徴量 vs 新規特徴量追加版を学習・比較評価
    """
    print("\n" + "=" * 75, flush=True)
    print("  🚀 LightGBM モデル精度比較: ベースライン vs 環境クロス＋代替モメンタム", flush=True)
    print("=" * 75, flush=True)
    print(f"  データ分割基準日          : {split_date} (Train: 過去〜2025年末, Test: 2026年〜)", flush=True)

    # 特徴量リスト定義
    baseline_features = [
        'boat_number',
        'racer_rank_num',
        'age',
        'weight',
        'nat_win_rate',
        'nat_quinella_rate',
        'loc_win_rate',
        'loc_quinella_rate',
        'motor_rate',
        'boat_rate',
        'wind_speed',
        'wave_height',
        'exhibition_time'
    ]

    new_cross_features = [
        # 風速クロス
        'makuri_rate',
        'makurizashi_rate',
        'wind_makuri_cross',
        'strong_wind_makuri',
        'wind_makurizashi_cross',
        'strong_wind_outer_adv',
        'wind_nige_vulnerability',
        # 波高クロス
        'wave_weight_prod',
        'wave_weight_ratio',
        'high_wave_heavy_penalty',
        'high_wave_inner_risk',
        # 代替モメンタム
        'ex_diff_from_race_min',
        'ex_diff_from_race_mean',
        'ex_rank_in_race',
        'ex_momentum_diff',
        'ex_momentum_deviation'
    ]

    enhanced_features = baseline_features + new_cross_features

    # データ分割
    train_mask = df['race_date'] < split_date
    test_mask = df['race_date'] >= split_date

    # 1着が確定している有効レコードのみ
    valid_mask = df['finish_order'].notna() & (df['finish_order'] > 0)
    train_df = df[train_mask & valid_mask].copy()
    test_df = df[test_mask & valid_mask].copy()

    print(f"  学習データ (Train) レコード数: {len(train_df):,} 行 ({len(train_df)//6:,} レース)")
    print(f"  検証データ (Test)  レコード数: {len(test_df):,} 行 ({len(test_df)//6:,} レース)")
    print("-" * 75, flush=True)

    y_train = train_df['is_win'].values
    y_test = test_df['is_win'].values

    lgb_params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'min_child_samples': 20,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    # ----------------------------------------------------
    # 1. ベースラインモデルの学習
    # ----------------------------------------------------
    print("  [Model 1/2] ベースラインモデル (従来特徴量 13個) を学習中...", flush=True)
    X_train_base = train_df[baseline_features]
    X_test_base = test_df[baseline_features]

    train_data_base = lgb.Dataset(X_train_base, label=y_train)
    test_data_base = lgb.Dataset(X_test_base, label=y_test, reference=train_data_base)

    evals_result_base = {}
    model_base = lgb.train(
        lgb_params,
        train_data_base,
        num_boost_round=300,
        valid_sets=[train_data_base, test_data_base],
        valid_names=['train', 'test'],
        callbacks=[lgb.record_evaluation(evals_result_base), lgb.early_stopping(50, verbose=False)]
    )

    preds_base = model_base.predict(X_test_base)
    
    # 評価指標算出
    auc_base = roc_auc_score(y_test, preds_base)
    logloss_base = log_loss(y_test, preds_base)
    brier_base = brier_score_loss(y_test, preds_base)
    
    # レース単位 Top-1 的中精度 (最も予測確率が高い艇が1着になった割合)
    test_df['pred_base'] = preds_base
    top1_base_correct = test_df.groupby('race_id').apply(
        lambda g: g.loc[g['pred_base'].idxmax(), 'is_win'] == 1
    ).mean()

    # ----------------------------------------------------
    # 2. 新規特徴量追加モデルの学習
    # ----------------------------------------------------
    print("  [Model 2/2] 新規特徴量追加モデル (環境クロス＋モメンタム 29個) を学習中...", flush=True)
    X_train_enh = train_df[enhanced_features]
    X_test_enh = test_df[enhanced_features]

    train_data_enh = lgb.Dataset(X_train_enh, label=y_train)
    test_data_enh = lgb.Dataset(X_test_enh, label=y_test, reference=train_data_enh)

    evals_result_enh = {}
    model_enh = lgb.train(
        lgb_params,
        train_data_enh,
        num_boost_round=300,
        valid_sets=[train_data_enh, test_data_enh],
        valid_names=['train', 'test'],
        callbacks=[lgb.record_evaluation(evals_result_enh), lgb.early_stopping(50, verbose=False)]
    )

    preds_enh = model_enh.predict(X_test_enh)

    # 評価指標算出
    auc_enh = roc_auc_score(y_test, preds_enh)
    logloss_enh = log_loss(y_test, preds_enh)
    brier_enh = brier_score_loss(y_test, preds_enh)

    test_df['pred_enh'] = preds_enh
    top1_enh_correct = test_df.groupby('race_id').apply(
        lambda g: g.loc[g['pred_enh'].idxmax(), 'is_win'] == 1
    ).mean()

    # ----------------------------------------------------
    # 3. 精度比較レポート出力
    # ----------------------------------------------------
    print("\n" + "=" * 75, flush=True)
    print("  📊 予測精度比較サマリー (Out-of-Time Test Set: 2026年〜)", flush=True)
    print("=" * 75, flush=True)
    print(f"  指標 (Metric)             | ベースライン (従来) | 新規特徴量追加版    | 改善効果 (差分)")
    print(f"  --------------------------+---------------------+---------------------+-----------------")
    print(f"  ROC-AUC (識別能力)        | {auc_base:>17.5f}   | {auc_enh:>17.5f}   | {auc_enh - auc_base:>+15.5f} {'(向上)' if auc_enh > auc_base else ''}")
    print(f"  LogLoss (交差エントロピー)| {logloss_base:>17.5f}   | {logloss_enh:>17.5f}   | {logloss_enh - logloss_base:>+15.5f} {'(改善)' if logloss_enh < logloss_base else ''}")
    print(f"  Brier Score (較正度)      | {brier_base:>17.5f}   | {brier_enh:>17.5f}   | {brier_enh - brier_base:>+15.5f} {'(改善)' if brier_enh < brier_base else ''}")
    print(f"  レース Top-1 予想的中率   | {top1_base_correct:>16.2%}   | {top1_enh_correct:>16.2%}   | {top1_enh_correct - top1_base_correct:>+14.2%} pt")
    print("=" * 75, flush=True)

    # ----------------------------------------------------
    # 4. Feature Importance 出力
    # ----------------------------------------------------
    importance_gain = model_enh.feature_importance(importance_type='gain')
    importance_split = model_enh.feature_importance(importance_type='split')
    total_gain = sum(importance_gain)

    feat_imp_df = pd.DataFrame({
        'Feature': enhanced_features,
        'Type': ['新規(環境/モメンタム)' if f in new_cross_features else '従来(ベースライン)' for f in enhanced_features],
        'Gain': importance_gain,
        'Gain_Ratio (%)': (importance_gain / total_gain) * 100.0,
        'Split_Count': importance_split
    }).sort_values(by='Gain', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 75, flush=True)
    print("  🏆 Feature Importance ランキング (Gain 寄与度順 Top 20)", flush=True)
    print("=" * 75, flush=True)
    print(f"  Rank | Feature Name                 | Category           | Gain Ratio | Split Count")
    print(f"  -----+------------------------------+--------------------+------------+------------")
    for i, row in feat_imp_df.head(20).iterrows():
        is_new_mark = "🌟" if row['Type'].startswith('新規') else "  "
        print(f"  {i+1:>4d} | {row['Feature']:<28} | {row['Type']:<18} | {row['Gain_Ratio (%)']:>9.2f}% | {row['Split_Count']:>10d} {is_new_mark}")
    print("-" * 75, flush=True)

    # 新規特徴量の合計Gain寄与度
    new_gain_sum = feat_imp_df[feat_imp_df['Type'].startswith('新規')]['Gain_Ratio (%)'].sum()
    print(f"  🌟 新規環境クロス・モメンタム特徴量の総合寄与度: {new_gain_sum:.2f}%")
    print("=" * 75 + "\n", flush=True)

    return {
        'auc_base': auc_base,
        'auc_enh': auc_enh,
        'logloss_base': logloss_base,
        'logloss_enh': logloss_enh,
        'top1_base': top1_base_correct,
        'top1_enh': top1_enh_correct,
        'feat_imp_df': feat_imp_df
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment Cross Features with Local SQLite")
    parser.add_argument('--start_date', type=str, default='2024-01-01', help="Start date for training data (default: 2024-01-01)")
    parser.add_argument('--split_date', type=str, default='2026-01-01', help="Split date for Out-of-Time test set (default: 2026-01-01)")
    parser.add_argument('--limit_races', type=int, default=50000, help="Max races to extract from SQLite (default: 50,000)")
    
    args = parser.parse_args()
    
    db_file = get_sqlite_path()
    raw_data = extract_raw_data(db_file, start_date=args.start_date, limit_races=args.limit_races)
    engineered_data = engineer_features(raw_data)
    train_and_compare_models(engineered_data, split_date=args.split_date)
