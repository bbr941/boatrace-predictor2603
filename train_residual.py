"""
train_residual.py
オッズ残差学習モデル (Phase 2 Extractor) の高度化学習スクリプト
LightGBM binary objective と init_score (ベースマージン) によるオッズ市場予測残差学習
75カラム拡張データセット対応・過学習抑制正則化・Out-of-Time評価
"""

import os
import sys
import argparse
import time
import shutil
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

import train_model
from odds_normalizer import probs_to_init_scores, odds_to_normalized_probs

# デフォルト設定
DEFAULT_DATA_PATH = 'train_data_full.csv' if os.path.exists('train_data_full.csv') else 'boatrace_dataset_labeled_v2.csv'
DEFAULT_MODEL_OUTPUT = 'model_residual.txt'
DEFAULT_MODEL_BACKUP = 'model_residual_backup.txt'


def load_and_preprocess_dataset(
    data_path: str = DEFAULT_DATA_PATH,
    sample_races: int = None,
    filter_valid_odds: bool = True
) -> pd.DataFrame:
    """
    データセットを読み込み、特徴量前処理および init_score (ベースマージン) を生成する。
    """
    print("=" * 75, flush=True)
    print(f"  📂 データセット読み込み中: {data_path}", flush=True)
    print("=" * 75, flush=True)
    t0 = time.time()

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"データセットが見つかりません: {data_path}")

    # 高速読み込み
    if sample_races is not None and sample_races > 0:
        print(f"  - サンプルモード: 上限 {sample_races} レースを抽出中...", flush=True)
        chunks = []
        collected_races = set()
        for chunk in pd.read_csv(data_path, chunksize=150000):
            if filter_valid_odds and 'syn_win_rate' in chunk.columns:
                valid_chunk = chunk[chunk['syn_win_rate'] > 0]
            else:
                valid_chunk = chunk
                
            new_races = valid_chunk['race_id'].unique()
            for r in new_races:
                collected_races.add(r)
                if len(collected_races) >= sample_races:
                    break
            chunks.append(valid_chunk[valid_chunk['race_id'].isin(collected_races)])
            if len(collected_races) >= sample_races:
                break
        df = pd.concat(chunks, ignore_index=True)
    else:
        print("  - 全件モード: データセットを直接ロード中...", flush=True)
        df = pd.read_csv(data_path)

    print(f"  読み込み完了: {len(df):,} 行 ({time.time() - t0:.2f} 秒)", flush=True)

    # 1. 特徴量エンジニアリング & 前処理
    df = train_model.preprocess_data(df)

    # 2. オッズの正規化と init_score (ベースマージン・ロジット) の生成
    print("  - オッズデータの正規化およびベースマージン (init_score) 算出中...", flush=True)
    if 'syn_win_rate' not in df.columns:
        df['syn_win_rate'] = 0.0

    df['syn_win_rate'] = pd.to_numeric(df['syn_win_rate'], errors='coerce').fillna(0.0)

    # レースごとに syn_win_rate の和を計算して正規化
    race_sums = df.groupby('race_id')['syn_win_rate'].transform('sum')
    has_valid_odds = (race_sums > 0) & np.isfinite(race_sums)
    
    # 控除率を排除した正規化確率 P_norm (和=1.0)
    p_norm = np.where(has_valid_odds, df['syn_win_rate'] / np.maximum(race_sums, 1e-9), 1.0 / 6.0)
    # 微小クリッピングとロジット変換
    init_scores = probs_to_init_scores(p_norm, clip_eps=1e-5)

    df['market_p_norm'] = p_norm
    df['init_score'] = init_scores
    df['has_valid_odds'] = has_valid_odds

    # 3. ターゲット変数をバイナリ (1着=1, 2〜6着=0) に変換
    df['target_binary'] = (df['rank'] == 1).astype(int)

    # オッズ残差学習のため、オッズが存在する有効レースに絞り込む
    if filter_valid_odds:
        n_before = len(df)
        df = df[df['has_valid_odds']].copy()
        print(f"  - オッズ有効レースにフィルタリング: {n_before:,} 行 -> {len(df):,} 行 ({df['race_id'].nunique():,} レース)", flush=True)

    return df


def get_residual_features(df: pd.DataFrame) -> list:
    """
    オッズ残差モデル用の特徴量リストを取得する。
    重要: オッズや確定配当に直接由来する変数はデータ漏洩 (リーク) 防止のため必ず厳密に除外する。
    """
    # 厳格なリーク・識別子除外リスト
    forbidden_exact = {
        'race_id', 'boat_number', 'racer_id', 'rank', 'relevance',
        'race_date', 'venue_name', 'racer_rank', 'prior_results',
        'weight_for_loss', 'pred_score', 'is_F_holder', 'temp_venue_code',
        'my_nige_rate', 'my_sashi_rate', 'my_makuri_rate', 'st_rank',
        'venue_code_x', 'venue_code_int', 'ana_relevance', 'weight_ana', 'proxy_odds',
        'weather', 'nige_count', 'makuri_count', 'makurizashi_count', 'sashi_count',
        'wintech_races_run', 'wintech_wins',
        # オッズ・配当・残差ベースマージン
        'syn_win_rate', 'odds', 'odds_1min', 'prediction_odds', 'popularity',
        'vote_count', 'win_share', 'init_score', 'market_p_norm', 'has_valid_odds',
        'target_binary', 'payout', 'payoff', 'profit', 'actual_result', 'is_resolved'
    }
    
    forbidden_substrings = ['odds', 'win_share', 'vote', 'popularity', 'payoff', 'payout']
    
    all_cols = df.columns.tolist()
    final_feats = []
    
    for c in all_cols:
        if c in forbidden_exact:
            continue
        if any(sub in c.lower() for sub in forbidden_substrings):
            continue
        final_feats.append(c)
            
    return sorted(list(set(final_feats)))


def train_residual_model(
    df: pd.DataFrame,
    features: list,
    model_output_path: str = DEFAULT_MODEL_OUTPUT,
    backup_path: str = DEFAULT_MODEL_BACKUP,
    split_date: str = '2026-01-01',
    learning_rate: float = 0.03,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50
):
    """
    LightGBM によるオッズ残差学習を実行する (Out-of-Time 分割 & 正則化)。
    """
    print("\n" + "=" * 75, flush=True)
    print("  🚀 LightGBM オッズ残差学習モデル (Phase 2 Extractor) のトレーニング開始", flush=True)
    print(f"  特徴量数: {len(features)} 件 | 目的関数: binary (binary_logloss) | init_score: 適用", flush=True)
    print("=" * 75, flush=True)

    # データをレース順にソート
    df = df.sort_values(['race_date', 'race_id', 'boat_number']).reset_index(drop=True)

    # 時系列分割 (Out-of-Time: split_date 基準)
    unique_dates = sorted(df['race_date'].dropna().unique())
    if split_date is None or df['race_date'].max() < split_date or df['race_date'].min() >= split_date:
        split_idx = int(len(unique_dates) * 0.8)
        effective_split_date = unique_dates[split_idx]
    else:
        effective_split_date = split_date

    train_mask = df['race_date'] < effective_split_date
    test_mask = df['race_date'] >= effective_split_date

    train_df = df[train_mask].copy()
    val_df = df[test_mask].copy()

    print(f"  実効データ分割基準日      : {effective_split_date} (Train: ~{effective_split_date}前日, Test: {effective_split_date}~)", flush=True)
    print(f"  学習データ (Train) レコード数: {len(train_df):,} 行 ({train_df['race_id'].nunique():,} レース)", flush=True)
    print(f"  検証データ (Test)  レコード数: {len(val_df):,} 行 ({val_df['race_id'].nunique():,} レース)", flush=True)
    print("-" * 75, flush=True)

    X_train = train_df[features]
    y_train = train_df['target_binary'].to_numpy()
    init_train = train_df['init_score'].to_numpy()

    X_val = val_df[features]
    y_val = val_df['target_binary'].to_numpy()
    init_val = val_df['init_score'].to_numpy()

    # LightGBM Dataset 構築 (init_score を渡す)
    print("  - LightGBM Dataset を構築中...", flush=True)
    dtrain = lgb.Dataset(X_train, label=y_train, init_score=init_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, init_score=init_val, reference=dtrain, free_raw_data=False)

    # 過学習抑制ハイパーパラメータ (正則化 & 深さ制限)
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': learning_rate,
        'max_depth': 6,                 # 正則化: 深さ制限 (過度な複雑分岐をカット)
        'num_leaves': 31,
        'min_data_in_leaf': 50,         # 正則化: 最小リーフデータ数
        'feature_fraction': 0.8,        # 特徴量サブサンプリング
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 1.0,               # L1正則化 (ノイズ特徴量の係数圧縮)
        'lambda_l2': 2.0,               # L2正則化 (極端な残差予測を抑制)
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    # バックアップ作成
    if os.path.exists(model_output_path):
        shutil.copyfile(model_output_path, backup_path)
        print(f"  旧モデルをバックアップ: {backup_path}", flush=True)

    print("  - 学習を開始します (num_boost_round=1000, early_stopping=50)...", flush=True)
    t0 = time.time()
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=50)
        ]
    )
    train_time = time.time() - t0
    print(f"\n  学習完了: {train_time:.2f} 秒 (Best Iteration: {model.best_iteration})", flush=True)

    # モデル保存
    model.save_model(model_output_path)
    print(f"  💾 モデル保存完了: {model_output_path} (サイズ: {os.path.getsize(model_output_path)/1024:.1f} KB)", flush=True)

    # ==========================================
    # 評価とパフォーマンス分析 (Out-of-Time)
    # ==========================================
    print("\n" + "=" * 75, flush=True)
    print(f"  📊 Out-of-Time テストデータ ({effective_split_date}〜 {val_df['race_id'].nunique():,}レース) 性能評価", flush=True)
    print("=" * 75, flush=True)

    # 1. ベースライン (オッズ市場予測のみ)
    p_market_val = val_df['market_p_norm'].to_numpy()
    p_market_clipped = np.clip(p_market_val, 1e-5, 1.0 - 1e-5)
    base_logloss = log_loss(y_val, p_market_clipped)
    base_brier = brier_score_loss(y_val, p_market_clipped)
    base_auc = roc_auc_score(y_val, p_market_clipped)

    # 2. 新残差モデル予測値 (生マージン + init_score -> シグモイド)
    raw_preds_new = model.predict(X_val, raw_score=True)
    margin_new = raw_preds_new + init_val
    p_new = 1.0 / (1.0 + np.exp(-margin_new))
    p_new_clipped = np.clip(p_new, 1e-5, 1.0 - 1e-5)

    new_logloss = log_loss(y_val, p_new_clipped)
    new_brier = brier_score_loss(y_val, p_new_clipped)
    new_auc = roc_auc_score(y_val, p_new_clipped)

    # 3. 旧残差モデル予測値 (バックアップが存在する場合)
    old_logloss, old_brier, old_auc, old_hit_rate = None, None, None, None
    if os.path.exists(backup_path):
        try:
            old_model = lgb.Booster(model_file=backup_path)
            old_feats = old_model.feature_name()
            if all(f in val_df.columns for f in old_feats):
                raw_preds_old = old_model.predict(val_df[old_feats], raw_score=True)
                margin_old = raw_preds_old + init_val
                p_old = 1.0 / (1.0 + np.exp(-margin_old))
                p_old_clipped = np.clip(p_old, 1e-5, 1.0 - 1e-5)
                old_logloss = log_loss(y_val, p_old_clipped)
                old_brier = brier_score_loss(y_val, p_old_clipped)
                old_auc = roc_auc_score(y_val, p_old_clipped)
                
                val_df['pred_p_old'] = p_old
                top_old_idx = val_df.groupby('race_id')['pred_p_old'].idxmax()
                top_old_boat = val_df.loc[top_old_idx].set_index('race_id')['boat_number']
        except Exception as e:
            print(f"  (旧モデル比較スキップ: {e})", flush=True)

    # Top-1 的中率
    val_df['pred_p_new'] = p_new
    actual_1st = val_df[val_df['rank'] == 1].set_index('race_id')['boat_number']
    
    top_mkt_idx = val_df.groupby('race_id')['market_p_norm'].idxmax()
    top_mkt_boat = val_df.loc[top_mkt_idx].set_index('race_id')['boat_number']
    
    top_new_idx = val_df.groupby('race_id')['pred_p_new'].idxmax()
    top_new_boat = val_df.loc[top_new_idx].set_index('race_id')['boat_number']

    common_races = actual_1st.index.intersection(top_mkt_boat.index).intersection(top_new_boat.index)
    total_races = len(common_races)

    mkt_hit_rate = (top_mkt_boat.loc[common_races] == actual_1st.loc[common_races]).mean()
    new_hit_rate = (top_new_boat.loc[common_races] == actual_1st.loc[common_races]).mean()
    if old_logloss is not None:
        old_hit_rate = (top_old_boat.loc[common_races] == actual_1st.loc[common_races]).mean()

    print(f"  【評価対象レース数】 : {total_races:,} レース ({len(val_df):,} 行)", flush=True)
    print(f"  ┌────────────────────────┬──────────────────┬──────────────────┬──────────────────┬──────────────┐", flush=True)
    print(f"  │ 評価メトリクス         │ オッズ単体(Base) │ 旧残差モデル     │ 新残差モデル     │ 改善効果 (Δ) │", flush=True)
    print(f"  ├────────────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────┤", flush=True)
    old_ll_str = f"{old_logloss:.5f}" if old_logloss is not None else "-"
    old_br_str = f"{old_brier:.5f}" if old_brier is not None else "-"
    old_auc_str = f"{old_auc:.5f}" if old_auc is not None else "-"
    old_hit_str = f"{old_hit_rate:7.2%}" if old_hit_rate is not None else "-"
    
    ll_diff = new_logloss - (old_logloss if old_logloss is not None else base_logloss)
    br_diff = new_brier - (old_brier if old_brier is not None else base_brier)
    auc_diff = new_auc - (old_auc if old_auc is not None else base_auc)
    hit_diff = new_hit_rate - (old_hit_rate if old_hit_rate is not None else mkt_hit_rate)

    print(f"  │ Binary LogLoss (低い程良)│     {base_logloss:.5f}      │     {old_ll_str:>8}     │     {new_logloss:.5f}      │ {ll_diff:+.5f} ({'改善' if ll_diff < 0 else '維持'})│", flush=True)
    print(f"  │ Brier Score    (低い程良)│     {base_brier:.5f}      │     {old_br_str:>8}     │     {new_brier:.5f}      │ {br_diff:+.5f} ({'改善' if br_diff < 0 else '維持'})│", flush=True)
    print(f"  │ ROC-AUC        (高い程良)│     {base_auc:.5f}      │     {old_auc_str:>8}     │     {new_auc:.5f}      │ {auc_diff:+.5f} ({'向上' if auc_diff > 0 else '維持'})│", flush=True)
    print(f"  │ Top-1 予想的中率 (1着) │     {mkt_hit_rate:7.2%}      │     {old_hit_str:>8}     │     {new_hit_rate:7.2%}      │ {hit_diff:+.2%} pt   │", flush=True)
    print(f"  └────────────────────────┴──────────────────┴──────────────────┴──────────────────┴──────────────┘", flush=True)

    # 4. 特徴量重要度 Top 25 (Gain)
    importances = model.feature_importance(importance_type='gain')
    splits = model.feature_importance(importance_type='split')
    total_gain = sum(importances)

    new_cross_features = {
        'wind_makuri_cross', 'strong_wind_makuri', 'wind_makurizashi_cross',
        'strong_wind_outer_adv', 'wind_nige_vulnerability',
        'wave_weight_prod', 'wave_weight_ratio', 'high_wave_heavy_penalty', 'high_wave_inner_risk',
        'ex_diff_from_race_min', 'ex_diff_from_race_mean', 'ex_rank_in_race',
        'ex_momentum_diff', 'ex_momentum_deviation', 'makurizashi_rate',
        'is_strong_wind', 'is_gale_wind', 'is_high_wave'
    }

    feat_imp = pd.DataFrame({
        'feature': features,
        'category': ['🌟 新規(クロス/モメンタム)' if f in new_cross_features else '従来(ベースライン)' for f in features],
        'gain': importances,
        'gain_ratio': (importances / total_gain) * 100.0,
        'split': splits
    }).sort_values('gain', ascending=False).reset_index(drop=True)

    print("\n" + "=" * 75, flush=True)
    print("  🏆 Feature Importance ランキング (Gain 寄与度順 Top 25)", flush=True)
    print("=" * 75, flush=True)
    print(f"  Rank | Feature Name                 | Category                | Gain Ratio | Split Count", flush=True)
    print(f"  -----+------------------------------+-------------------------+------------+------------", flush=True)
    for rank, row in feat_imp.head(25).iterrows():
        is_new_mark = "🌟" if "新規" in str(row['category']) else "  "
        print(f"  {rank+1:4d} | {row['feature']:<28} | {row['category']:<23} | {row['gain_ratio']:>9.2f}% | {row['split']:>10d} {is_new_mark}", flush=True)
    print("-" * 75, flush=True)

    new_gain_sum = feat_imp[feat_imp['category'].str.contains('新規')]['gain_ratio'].sum()
    print(f"  🌟 新規環境クロス・モメンタム特徴量の残差抽出寄与度: {new_gain_sum:.2f}%", flush=True)
    print("=" * 75 + "\n", flush=True)

    return model, feat_imp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Odds Residual Model for Boatrace (Phase 2 Extractor)")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument('--output', type=str, default=DEFAULT_MODEL_OUTPUT, help="Output path for trained model")
    parser.add_argument('--split_date', type=str, default='2026-01-01', help="Out-of-Time split date")
    parser.add_argument('--sample_races', type=int, default=None, help="Sample N races for fast training")
    parser.add_argument('--lr', type=float, default=0.03, help="Learning rate (default: 0.03)")
    parser.add_argument('--rounds', type=int, default=1000, help="Max boost rounds (default: 1000)")
    parser.add_argument('--early_stopping', type=int, default=50, help="Early stopping rounds (default: 50)")

    args = parser.parse_args()

    # 1. データセット読み込み
    df = load_and_preprocess_dataset(data_path=args.data_path, sample_races=args.sample_races)

    # 2. 特徴量抽出 (オッズ由来列を厳格に除外)
    features = get_residual_features(df)
    print(f"  使用特徴量 ({len(features)} 件): {features}\n")

    # 3. 学習 & 評価
    train_residual_model(
        df=df,
        features=features,
        model_output_path=args.output,
        split_date=args.split_date,
        learning_rate=args.lr,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping
    )

