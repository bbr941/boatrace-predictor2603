"""
train_residual.py
オッズ残差学習モデル (Phase 2) の新規学習スクリプト
LightGBM binary objective と init_score (ベースマージン) によるオッズ市場予測残差学習
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import log_loss, roc_auc_score

import train_model
from odds_normalizer import probs_to_init_scores, odds_to_normalized_probs

# デフォルト設定
DEFAULT_DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
DEFAULT_MODEL_OUTPUT = 'model_residual.txt'


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

    # 大規模CSVの高速チャンク読み込み
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
        print("  - 全件モード: オッズ有効レコードを高速チャンク抽出中...", flush=True)
        chunks = []
        for chunk in pd.read_csv(data_path, chunksize=150000):
            if filter_valid_odds and 'syn_win_rate' in chunk.columns:
                valid_chunk = chunk[chunk['syn_win_rate'] > 0]
            else:
                valid_chunk = chunk
            if len(valid_chunk) > 0:
                chunks.append(valid_chunk)
        df = pd.concat(chunks, ignore_index=True)

    print(f"  読み込み完了: {len(df):,} 行 ({time.time() - t0:.2f} 秒)", flush=True)

    # 1. 特徴量エンジニアリング & 前処理
    df = train_model.preprocess_data(df)

    # 2. オッズの正規化と init_score (ベースマージン・ロジット) の生成
    print("  - オッズデータの正規化およびベースマージン (init_score) 算出中...", flush=True)
    if 'syn_win_rate' not in df.columns:
        df['syn_win_rate'] = 0.0

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

    # オッズ残差学習のため、オッズが存在するレースに絞り込む (オプション)
    if filter_valid_odds:
        n_before = len(df)
        df = df[df['has_valid_odds']].copy()
        print(f"  - オッズ有効レースにフィルタリング: {n_before:,} 行 -> {len(df):,} 行 ({df['race_id'].nunique():,} レース)", flush=True)

    return df


def get_residual_features(df: pd.DataFrame) -> list:
    """
    オッズ残差モデル用の特徴量リストを取得する。
    重要: オッズに直接由来する変数 (syn_win_rate, odds等) はデータ漏洩防止のため必ず除外する。
    """
    # train_model.get_features(df, mode='ana') をベースとし、追加で不要列を除外
    feats = train_model.get_features(df, mode='ana')
    
    forbidden_keywords = ['syn_win_rate', 'odds', 'popularity', 'vote_count', 'win_share', 'init_score', 'market_p_norm', 'has_valid_odds', 'target_binary']
    final_feats = []
    for f in feats:
        if not any(k in f.lower() for k in forbidden_keywords):
            final_feats.append(f)
            
    return sorted(list(set(final_feats)))


def train_residual_model(
    df: pd.DataFrame,
    features: list,
    model_output_path: str = DEFAULT_MODEL_OUTPUT,
    learning_rate: float = 0.03,
    num_boost_round: int = 1000,
    early_stopping_rounds: int = 50
):
    """
    LightGBM によるオッズ残差学習を実行する。
    """
    print("\n" + "=" * 75, flush=True)
    print("  🚀 LightGBM オッズ残差学習モデル (Phase 2) のトレーニング開始", flush=True)
    print(f"  特徴量数: {len(features)} 件 | 目的関数: binary (binary_logloss) | init_score: 適用", flush=True)
    print("=" * 75, flush=True)

    # 時系列分割 (過去 80% Train, 直近 20% Validation/Test)
    unique_races = df['race_id'].unique()
    split_idx = int(len(unique_races) * 0.8)
    train_races_set = set(unique_races[:split_idx])

    train_mask = df['race_id'].isin(train_races_set)
    train_df = df[train_mask]
    val_df = df[~train_mask].copy()

    n_train_races = split_idx
    n_val_races = len(unique_races) - split_idx

    print(f"  Train データ: {len(train_df):,} 行 ({n_train_races:,} レース)", flush=True)
    print(f"  Val   データ: {len(val_df):,} 行 ({n_val_races:,} レース)", flush=True)

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

    # ハイパーパラメータ設定
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'learning_rate': learning_rate,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }

    print("  - 学習を開始します (num_boost_round=1000, early_stopping=50)...", flush=True)
    t0 = time.time()
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        num_boost_round=num_boost_round,
        callbacks=[
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    train_time = time.time() - t0
    print(f"\n  学習完了: {train_time:.2f} 秒 (Best Iteration: {model.best_iteration})", flush=True)

    # モデル保存
    model.save_model(model_output_path)
    print(f"  💾 モデル保存完了: {model_output_path}", flush=True)

    # ==========================================
    # 評価とパフォーマンス分析
    # ==========================================
    print("\n" + "=" * 75, flush=True)
    print("  📊 テストデータ (直近20%) における性能評価レポート", flush=True)
    print("=" * 75, flush=True)

    # 1. ベースライン (オッズ市場予測のみの LogLoss)
    p_market_val = val_df['market_p_norm'].to_numpy()
    p_market_clipped = np.clip(p_market_val, 1e-5, 1.0 - 1e-5)
    baseline_logloss = log_loss(y_val, p_market_clipped)
    baseline_auc = roc_auc_score(y_val, p_market_clipped)

    # 2. オッズ残差モデルの予測値 (生マージン + init_score -> シグモイド)
    raw_preds = model.predict(X_val, raw_score=True)
    total_margin = raw_preds + init_val
    p_pred = 1.0 / (1.0 + np.exp(-total_margin))
    p_pred_clipped = np.clip(p_pred, 1e-5, 1.0 - 1e-5)

    model_logloss = log_loss(y_val, p_pred_clipped)
    model_auc = roc_auc_score(y_val, p_pred_clipped)

    # 3. 1着的中精度 (レース内で最高確率の艇が1着となった割合) - 高速ベクトル化処理
    val_df['pred_prob'] = p_pred
    
    # 実際の1着艇
    actual_1st = val_df[val_df['rank'] == 1].set_index('race_id')['boat_number']
    
    # 市場予測および残差モデルの予測1着艇
    top_market_idx = val_df.groupby('race_id')['market_p_norm'].idxmax()
    top_market_boat = val_df.loc[top_market_idx].set_index('race_id')['boat_number']
    
    top_model_idx = val_df.groupby('race_id')['pred_prob'].idxmax()
    top_model_boat = val_df.loc[top_model_idx].set_index('race_id')['boat_number']
    
    # 共通レースで比較
    common_races = actual_1st.index.intersection(top_market_boat.index).intersection(top_model_boat.index)
    total_val_races = len(common_races)
    
    market_hits = (top_market_boat.loc[common_races] == actual_1st.loc[common_races]).sum()
    model_hits = (top_model_boat.loc[common_races] == actual_1st.loc[common_races]).sum()

    market_hit_rate = market_hits / total_val_races if total_val_races > 0 else 0
    model_hit_rate = model_hits / total_val_races if total_val_races > 0 else 0
    logloss_diff = model_logloss - baseline_logloss

    print(f"  【評価対象レース数】 : {total_val_races:,} レース ({len(val_df):,} 行)", flush=True)
    print(f"  ┌────────────────────────┬──────────────────┬──────────────────┬──────────────┐", flush=True)
    print(f"  │ 評価メトリクス         │ オッズ単体(Base) │ 残差モデル(提案) │ 改善度 (Δ)   │", flush=True)
    print(f"  ├────────────────────────┼──────────────────┼──────────────────┼──────────────┤", flush=True)
    print(f"  │ Binary Log Loss (低い程良)│     {baseline_logloss:.5f}      │     {model_logloss:.5f}      │ {logloss_diff:+.5f} ({'改善' if logloss_diff < 0 else '維持'})│", flush=True)
    print(f"  │ ROC-AUC (高い程良)     │     {baseline_auc:.5f}      │     {model_auc:.5f}      │ {model_auc - baseline_auc:+.5f}        │", flush=True)
    print(f"  │ 1番人気/本命 1着的中率 │     {market_hit_rate:7.2%}      │     {model_hit_rate:7.2%}      │ {model_hit_rate - market_hit_rate:+.2%}        │", flush=True)
    print(f"  └────────────────────────┴──────────────────┴──────────────────┴──────────────┘", flush=True)

    # 4. 特徴量重要度 Top 15
    importances = model.feature_importance(importance_type='gain')
    feat_imp = pd.DataFrame({'feature': features, 'importance_gain': importances})
    feat_imp = feat_imp.sort_values('importance_gain', ascending=False).reset_index(drop=True)

    print("\n  【特徴量重要度 Top 15 (Gain)】", flush=True)
    for rank, row in feat_imp.head(15).iterrows():
        print(f"    {rank+1:2d}位: {row['feature']:<28} (Gain: {row['importance_gain']:,.1f})", flush=True)

    print("\n" + "=" * 75, flush=True)
    print("  🎉 オッズ残差学習モデルのトレーニングおよび評価が正常に完了しました！", flush=True)
    print("=" * 75 + "\n", flush=True)

    return model, feat_imp


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Odds Residual Model for Boatrace (Phase 2)")
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help="Path to CSV dataset")
    parser.add_argument('--output', type=str, default=DEFAULT_MODEL_OUTPUT, help="Output path for trained model")
    parser.add_argument('--sample_races', type=int, default=None, help="Sample N races for fast training")
    parser.add_argument('--lr', type=float, default=0.03, help="Learning rate (default: 0.03)")
    parser.add_argument('--rounds', type=int, default=1000, help="Max boost rounds (default: 1000)")
    parser.add_argument('--early_stopping', type=int, default=50, help="Early stopping rounds (default: 50)")

    args = parser.parse_args()

    # 1. データセット読み込み
    df = load_and_preprocess_dataset(data_path=args.data_path, sample_races=args.sample_races)

    # 2. 特徴量抽出 (オッズ由来列を除外)
    features = get_residual_features(df)
    print(f"  使用特徴量 ({len(features)} 件): {features}")

    # 3. 学習 & 評価
    train_residual_model(
        df=df,
        features=features,
        model_output_path=args.output,
        learning_rate=args.lr,
        num_boost_round=args.rounds,
        early_stopping_rounds=args.early_stopping
    )
