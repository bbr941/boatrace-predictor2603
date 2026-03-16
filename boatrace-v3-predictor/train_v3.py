import os
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
from tabpfn import TabPFNClassifier
import optuna
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import warnings

# --- TabPFN / PyTorch 2.6+ Compatibility Fix ---
# Python 3.13 / PyTorch 2.6+ restrict torch.load for security.
# TabPFN 0.1.9 relies on older serialization. Let's patch it.
original_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_torch_load
# -----------------------------------------------

# --- Configuration ---
ENABLE_L2_FEATURE_SELECTION = False # V3.1 logic flag

# --- V4 Experiment Flags ---
ENABLE_EMA_MOMENTUM = False      # Phase 4 (Rollback)
ENABLE_TIME_DECAY_WEIGHT = True  # Phase 5 (Adopted)
ENABLE_FM_MODEL = False          # Phase 6 (Rollback)

# --- V4 Final Experiment Flags (Phase 7-9) ---
ENABLE_NULL_IMPORTANCE = False   # Phase 7 (Rollback)
ENABLE_ODDS_WEIGHT = False       # Phase 8 (Rollback)
ENABLE_ROI_OPTUNA_META = True    # Phase 9
# ---------------------------
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# GPU Settings (AMD ROCm / General GPU)
# LightGBM CPU for stability with high-cardinality features on some ROCm versions
LGB_PARAMS_BASE = {
    'device': 'cpu',
    'verbosity': -1
}

CB_PARAMS_BASE = {
    'task_type': 'CPU', # Switched to CPU for max compatibility during debug
    'random_seed': 42,
    'verbose': 100
}

def load_data(limit=20000):
    """データベースから学習データをロードし、前回と同様の特徴量エンジニアリングを施す"""
    print(f"Loading latest {limit} rows from boatrace.db...")
    conn = sqlite3.connect(DB_PATH)
    
    # 既存の make_data_set.py のクエリをベースにする
    query = f"""
    SELECT
        re.race_id, re.boat_number, re.racer_id, r.venue_code, r.race_date,
        bi.exhibition_time, bi.exhibition_start_timing, 
        COALESCE(bi.exhibition_entry_course, re.boat_number) as pred_course,
        re.nat_win_rate, re.motor_rate, re.boat_rate, re.prior_results,
        res.finish_order as rank,
        pa.sanrentan_payoff
    FROM race_entries re
    JOIN races r ON re.race_id = r.race_id
    LEFT JOIN before_info bi ON re.race_id = bi.race_id AND re.boat_number = bi.boat_number
    LEFT JOIN results res ON re.race_id = res.race_id AND re.boat_number = res.boat_number
    LEFT JOIN payoffs pa ON re.race_id = pa.race_id
    WHERE res.finish_order IS NOT NULL
    ORDER BY r.race_date DESC, re.race_id DESC
    LIMIT {limit}
    """
    df = pd.read_sql(query, conn)
    conn.close()
    
    # --- Feature Engineering: Phase 1 Relative Features ---
    print("Adding Phase 1: Relative Features...")
    
    # 展示タイムの偏差値 (小さいほど良い)
    def zscore(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else 0

    df['exhibition_time_z'] = df.groupby('race_id')['exhibition_time'].transform(zscore)
    df['nat_win_rate_z'] = df.groupby('race_id')['nat_win_rate'].transform(zscore)
    
    # 1号艇との差分
    b1_win_rates = df[df['boat_number'] == 1][['race_id', 'nat_win_rate']].rename(columns={'nat_win_rate': 'b1_nat_win_rate'})
    df = df.merge(b1_win_rates, on='race_id', how='left')
    df['win_rate_diff_b1'] = df['nat_win_rate'] - df['b1_nat_win_rate'].fillna(df['nat_win_rate'])
    df.drop('b1_nat_win_rate', axis=1, inplace=True)
    
    # --- Feature Engineering: Phase 4 EMA Momentum ---
    if ENABLE_EMA_MOMENTUM:
        print("Adding Phase 4: EMA Momentum features (with shift(1))...")
        # Ensure time-series order for EACH racer
        # We use race_id as a proxy for time if date is not in the select (but ordering is by date in query)
        # Actually, let's sort by race_id globally first
        df_sorted = df.sort_values(['racer_id', 'race_id']).copy()
        
        # Calculate EMA for 'rank' and 'exhibition_time'
        # span=5 for recent trend
        # VERY IMPORTANT: shift(1) to avoid leakage (current race outcome must not be in features)
        df_sorted['ema_rank_5'] = df_sorted.groupby('racer_id')['rank'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
        df_sorted['ema_ex_time_5'] = df_sorted.groupby('racer_id')['exhibition_time'].transform(lambda x: x.ewm(span=5, adjust=False).mean().shift(1))
        
        # Merge back to original df
        df = df.merge(df_sorted[['race_id', 'boat_number', 'ema_rank_5', 'ema_ex_time_5']], 
                      on=['race_id', 'boat_number'], how='left')
        # Fill first races with 0 or neutral (rank 3.5 is average)
        df['ema_rank_5'] = df['ema_rank_5'].fillna(3.5)
        df['ema_ex_time_5'] = df['ema_ex_time_5'].fillna(df['exhibition_time'])
    
    # --- Standard Preprocessing ---
    # rank 1st -> 5, 2nd -> 4, ..., 6th -> 0 (for lambdarank relevance)
    df['rank_target'] = (6 - df['rank']).astype(int)
    df['target'] = (df['rank'] == 1).astype(int)
    
    # --- Feature Engineering: Phase 2 Target Encoding ---
    print("Adding Phase 2: K-Fold Target Encoding for racer_id...")
    from sklearn.model_selection import KFold
    df['racer_target_enc'] = 0.0
    kf = KFold(n_splits=5, shuffle=False)
    global_mean = df['target'].mean()
    df['racer_id_str'] = df['racer_id'].fillna(0).astype(int).astype(str)
    for train_idx, val_idx in kf.split(df):
        train_fold = df.iloc[train_idx]
        racer_stats = train_fold.groupby('racer_id_str')['target'].agg(['mean', 'count'])
        m = 10
        racer_stats['smoothed'] = (racer_stats['count'] * racer_stats['mean'] + m * global_mean) / (racer_stats['count'] + m)
        mapping = racer_stats['smoothed'].to_dict()
        df.loc[df.index[val_idx], 'racer_target_enc'] = df.loc[df.index[val_idx], 'racer_id_str'].map(mapping).fillna(global_mean)

    df['racer_id'] = df['racer_id'].fillna(0).astype(int).astype(str).astype('category')
    df['venue_code'] = df['venue_code'].fillna(0).astype(int).astype(str).astype('category')
    
    num_cols = ['exhibition_time', 'exhibition_start_timing', 'nat_win_rate', 'motor_rate', 'boat_rate', 
                'exhibition_time_z', 'nat_win_rate_z', 'win_rate_diff_b1', 'racer_target_enc']
    if ENABLE_EMA_MOMENTUM:
        num_cols.extend(['ema_rank_5', 'ema_ex_time_5'])

    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # --- Final Mapping for Inference (using ALL available data for the map) ---
    racer_stats = df.groupby('racer_id_str')['target'].agg(['mean', 'count'])
    # Smoothing m=10 (same as in K-Fold loop)
    m = 10
    racer_stats['smoothed'] = (racer_stats['count'] * racer_stats['mean'] + m * global_mean) / (racer_stats['count'] + m)
    racer_mapping = racer_stats['smoothed'].to_dict()
    racer_mapping['global_mean'] = global_mean

    return df, racer_mapping

def softmax_calibration(scores, group_sizes):
    probs = np.zeros_like(scores)
    idx = 0
    for size in group_sizes:
        if size == 0: continue
        s = scores[idx:idx+size]
        e_x = np.exp(s - np.max(s))
        probs[idx:idx+size] = e_x / e_x.sum()
        idx += size
    return probs

def train_lgb(X_train, y_train, X_val, y_val, train_groups, val_groups, cat_features, train_weights=None):
    print("\n--- Training LightGBM LambdaRank (CPU Mode) ---")
    def objective(trial):
        param = {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_at': [1, 3],
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.7, 1.0),
            **LGB_PARAMS_BASE
        }
        train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=train_weights, categorical_feature=cat_features)
        val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds, categorical_feature=cat_features)
        gbm = lgb.train(param, train_ds, valid_sets=[val_ds], 
                        callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(period=0)])
        return gbm.best_score['valid_0']['ndcg@1']

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5)
    
    best_params = study.best_params
    best_params.update(LGB_PARAMS_BASE)
    best_params['objective'] = 'lambdarank'
    
    train_ds = lgb.Dataset(X_train, label=y_train, group=train_groups, weight=train_weights, categorical_feature=cat_features)
    val_ds = lgb.Dataset(X_val, label=y_val, group=val_groups, reference=train_ds, categorical_feature=cat_features)
    model = lgb.train(best_params, train_ds, valid_sets=[val_ds], callbacks=[lgb.early_stopping(stopping_rounds=50)])
    return model

def train_catboost(X_train, y_train, X_val, y_val, cat_features, train_weights=None):
    print("\n--- Training CatBoost (CPU Mode) ---")
    model = cb.CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        early_stopping_rounds=50,
        **CB_PARAMS_BASE
    )
    model.fit(X_train, y_train, eval_set=(X_val, y_val), cat_features=cat_features, sample_weight=train_weights)
    return model

def train_alt_model(X_train, y_train, X_val, y_val, train_weights=None):
    """TabPFN の外部依存問題を回避するための代替基底モデル (Strong Regularized LR)"""
    print(f"\n--- Training Alternative Model (L2-Regularized LR) [Feature Selection: {ENABLE_L2_FEATURE_SELECTION}] ---")
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # 欠損値補完は load_data で済んでいる前提、スケーリングを行う
    scaler = StandardScaler()
    
    if ENABLE_L2_FEATURE_SELECTION:
        # 改良版: 線形モデルに相性の良い、順序に意味のある数値変数のみを厳選
        selected_features = [
            'exhibition_time', 'exhibition_start_timing', 'pred_course',
            'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_target_enc'
        ]
        X_train_sub = X_train[selected_features]
        X_val_sub = X_val[selected_features]
        print(f"  Selected Features for Alt-L2: {selected_features}")
    else:
        # 従来版 (V3.1): カテゴリ変数以外をすべて使用
        X_train_sub = X_train.drop(X_train.select_dtypes(['category']).columns, axis=1)
        X_val_sub = X_val.drop(X_val.select_dtypes(['category']).columns, axis=1)
        print(f"  Legacy Mode: Using all numerical columns: {list(X_train_sub.columns)}")
    
    X_train_scaled = scaler.fit_transform(X_train_sub)
    X_val_scaled = scaler.transform(X_val_sub)
    
    # 強力な正則化を施した LR
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train_scaled, y_train, sample_weight=train_weights)
    
    return model, X_val_scaled, scaler

# --- Phase 6: Factorization Machines (FM) Implementation ---
class SimpleFM(torch.nn.Module):
    def __init__(self, n_features, n_factors):
        super(SimpleFM, self).__init__()
        self.w0 = torch.nn.Parameter(torch.zeros(1))
        self.w = torch.nn.Embedding(n_features, 1)
        self.v = torch.nn.Embedding(n_features, n_factors)
        
        torch.nn.init.xavier_uniform_(self.v.weight)
        torch.nn.init.zeros_(self.w.weight)
        
    def forward(self, x):
        # x: [batch_size, n_input_features] indices
        linear_part = self.w0 + self.w(x).sum(dim=1).squeeze(1)
        
        # Interaction part: 0.5 * sum((sum(v_i * x_i)^2) - sum((v_i * x_i)^2))
        # Since x_i are indicators (all 1), it simplifies to:
        emb = self.v(x) # [batch_size, n_input_features, n_factors]
        sum_square = torch.pow(emb.sum(dim=1), 2)
        square_sum = torch.pow(emb, 2).sum(dim=1)
        interaction_part = 0.5 * (sum_square - square_sum).sum(dim=1)
        
        return torch.sigmoid(linear_part + interaction_part)

def train_fm(df_train, df_val, target_col='target'):
    print("\n--- Training Factorization Machine (PyTorch) ---")
    global ENABLE_FM_MODEL # Allow disabling on failure
    
    try:
        from sklearn.preprocessing import LabelEncoder
        fm_features = ['venue_code', 'racer_id', 'boat_number']
        
        # Prepare indices
        X_train_fm = pd.DataFrame()
        X_val_fm = pd.DataFrame()
        
        # Label Encoding across both sets to ensure same mapping
        combined = pd.concat([df_train[fm_features], df_val[fm_features]], axis=0)
        offset = 0
        total_dims = 0
        for col in fm_features:
            le = LabelEncoder()
            # Convert to str to handle potential mix of types
            combined_vals = le.fit_transform(combined[col].astype(str))
            X_train_fm[col] = combined_vals[:len(df_train)] + offset
            X_val_fm[col] = combined_vals[len(df_train):] + offset
            offset += len(le.classes_)
            total_dims += len(le.classes_)
            
        train_tensor = torch.tensor(X_train_fm.values, dtype=torch.long)
        val_tensor = torch.tensor(X_val_fm.values, dtype=torch.long)
        y_train_tensor = torch.tensor(df_train[target_col].values, dtype=torch.float32)
        y_val_tensor = torch.tensor(df_val[target_col].values, dtype=torch.float32)
        
        model = SimpleFM(total_dims, n_factors=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        # Training loop
        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(train_tensor)
            loss = criterion(outputs, y_train_tensor)
            
            if torch.isnan(loss):
                raise ValueError("NaN detected in FM loss")
                
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
                
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_tensor)
            fm_val_preds = val_outputs.numpy()
            
        print(f"  FM LogLoss: {log_loss(y_val_tensor, fm_val_preds):.4f}")
        return model, fm_val_preds, le # (Simplified le return for brevity)
        
    except Exception as e:
        print(f"  [CRITICAL] FM Training failed: {e}. Falling back to ENABLE_FM_MODEL = False.")
        ENABLE_FM_MODEL = False
        return None, None, None

def get_null_importance(X, y, groups, cat_features, iterations=10):
    """シャッフルラベル学習を繰り返し、特徴量のノイズ寄与分を算出する"""
    print(f"\n--- Calculating Null Importance ({iterations} iterations) ---")
    import copy
    
    actual_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'verbosity': -1,
        'device': 'cpu'
    }
    
    # 1. 実際の重要度
    train_ds = lgb.Dataset(X, label=y, group=groups, categorical_feature=cat_features)
    actual_model = lgb.train(actual_params, train_ds, num_boost_round=100)
    actual_imp = pd.Series(actual_model.feature_importance(importance_type='gain'), index=X.columns)
    
    # 2. Null 重要度の計算
    null_imps = []
    for i in range(iterations):
        y_shuffled = np.random.permutation(y)
        shuffled_ds = lgb.Dataset(X, label=y_shuffled, group=groups, categorical_feature=cat_features)
        null_model = lgb.train(actual_params, shuffled_ds, num_boost_round=100)
        null_imps.append(null_model.feature_importance(importance_type='gain'))
        if (i+1) % 5 == 0:
            print(f"  Null Importance Trial {i+1}/{iterations} done.")
            
    null_imp_df = pd.DataFrame(null_imps, columns=X.columns)
    
    # 特徴量ごとの「偽の貢献」しきい値 (75パーセンタイル)
    null_thresholds = null_imp_df.quantile(0.75)
    
    # 判定: 実際の重要度 < Null 重要度の閾値 なら削除
    features_to_drop = [col for col in X.columns if actual_imp[col] < null_thresholds[col]]
    
    print("\n--- Null Importance Results ---")
    for col in X.columns:
        status = "[DROP]" if col in features_to_drop else "[KEEP]"
        print(f"  {status} {col:25} | Actual: {actual_imp[col]:10.1f} | Null-Threshold: {null_thresholds[col]:10.1f}")
        
    return features_to_drop

def evaluate_roi(df_val, preds, threshold=0.1):
    """検証データでの簡易 ROI 計算 (1着単勝または3連単の期待値的アプローチ)"""
    val_data = df_val.copy()
    val_data['pred_prob'] = preds
    
    # 実際の結果と配当があるレースのみを対象にする
    # 単純化のため、各レースで最も予測確率の高い艇が1着になると予想
    top_preds = val_data.sort_values(['race_id', 'pred_prob'], ascending=[True, False]).groupby('race_id').head(1)
    
    # 実際的な単勝配当に近い値を算出 (payoff/15 程度)
    investment = 0
    total_returns = 0
    
    # しきい値以上のレースにのみ賭ける
    bet_races = top_preds[top_preds['pred_prob'] >= threshold]
    if len(bet_races) == 0:
        return 0.0
        
    investment = len(bet_races) * 100
    wins = bet_races[bet_races['rank'] == 1]
    returns = wins['sanrentan_payoff'].fillna(0).sum() / 15 
    
    actual_roi = (returns / investment) * 100 if investment > 0 else 0
    return actual_roi

def get_optimal_weights_roi(X_meta, y_val, df_val):
    """Optuna で平均 ROI を最大化するアンサンブル重みを探索する (Cross Validation)"""
    print("\n--- Optimizing Meta-Weights for ROI via Optuna (K-Fold CV) ---")
    import optuna
    from sklearn.model_selection import KFold
    
    # Disable optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        w1 = trial.suggest_float('w_lgb', 0, 1)
        w2 = trial.suggest_float('w_cb', 0, 1)
        w3 = trial.suggest_float('w_alt', 0, 1)
        
        s = w1 + w2 + w3 + 1e-6
        w1, w2, w3 = w1/s, w2/s, w3/s
        
        kf = KFold(n_splits=3, shuffle=False)
        rois = []
        
        for train_idx, test_idx in kf.split(X_meta):
            fold_preds = w1 * X_meta[test_idx, 0] + w2 * X_meta[test_idx, 1] + w3 * X_meta[test_idx, 2]
            fold_df = df_val.iloc[test_idx]
            roi = evaluate_roi(fold_df, fold_preds, threshold=0.15) # Evaluation threshold
            rois.append(roi)
            
        return np.mean(rois)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    best_w = study.best_params
    s = sum(best_w.values()) + 1e-6
    weights = {k: v/s for k, v in best_w.items()}
    print(f"  Optimal Weights: {weights}")
    print(f"  Best CV Mean ROI: {study.best_value:.2f}%")
    return weights

def main():
    df, racer_mapping = load_data(30000)
    
    features = ['venue_code', 'exhibition_time', 'exhibition_start_timing', 'pred_course', 
                'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_id',
                'exhibition_time_z', 'nat_win_rate_z', 'win_rate_diff_b1', 'racer_target_enc']
    
    if ENABLE_EMA_MOMENTUM:
        features.extend(['ema_rank_5', 'ema_ex_time_5'])
        print(f"  V4 Phase 4 Enabled: Added {['ema_rank_5', 'ema_ex_time_5']}")

    cat_features = ['venue_code', 'racer_id']
    target = 'target'
    rank_target = 'rank_target'
    
    # Ensure categorical features are strings/categories for consistency
    cat_features = ['venue_code', 'racer_id']
    for col in cat_features:
        df[col] = df[col].astype(str).astype('category')

    # --- Phase 7: Null Importance Feature Selection ---
    if ENABLE_NULL_IMPORTANCE:
        # We use a subset or full data for importance check
        # Shuffle for importance check
        temp_df = df.iloc[:10000].copy() # Speed up with subset
        temp_groups = temp_df.groupby('race_id', sort=False)['race_id'].count().values
        temp_X = temp_df[features]
        # For Null Importance, use rank_target as it's the LambdaRank target
        temp_y = temp_df[rank_target]
        
        # Identify categories that are in features
        temp_cat = [c for c in cat_features if c in features]
        
        to_drop = get_null_importance(temp_X, temp_y, temp_groups, temp_cat, iterations=10)
        features = [f for f in features if f not in to_drop]
        print(f"\nPhase 7: Dropped {len(to_drop)} redundant features. Final set size: {len(features)}")
        # Update cat_features as well
        cat_features = [c for c in cat_features if c in features]
    
    # Train/Val Split (Time-series split, shuffle=False)
    # Ensure races are not split between train/val by sorting correctly
    df = df.sort_values(['race_id', 'boat_number']).reset_index(drop=True)
    train_df, val_df = train_test_split(df, test_size=0.2, shuffle=False)
    
    # Calculate group sizes for LambdaRank
    train_groups = train_df.groupby('race_id', sort=False)['race_id'].count().values
    val_groups = val_df.groupby('race_id', sort=False)['race_id'].count().values
    
    X_train = train_df[features]
    y_train = train_df[target]
    y_train_rank = train_df[rank_target]
    
    X_val = val_df[features]
    y_val = val_df[target]
    y_val_rank = val_df[rank_target]
    
    # --- Phase 5 & 8: Sample Weights (Time Decay + Odds Weight) ---
    train_weights = None
    if ENABLE_TIME_DECAY_WEIGHT or ENABLE_ODDS_WEIGHT:
        print("\nCalculating Final Sample Weights...")
        train_df['race_date_dt'] = pd.to_datetime(train_df['race_date'])
        latest_date = train_df['race_date_dt'].max()
        train_df['days_diff'] = (latest_date - train_df['race_date_dt']).dt.days
        
        # Time Decay (Phase 5)
        w_time = np.exp(-0.001 * train_df['days_diff']) if ENABLE_TIME_DECAY_WEIGHT else np.ones(len(train_df))
        
        # Odds Weight (Phase 8)
        w_odds = np.ones(len(train_df))
        if ENABLE_ODDS_WEIGHT:
            # Use log(1 + payoff/100) and scale to avoid extreme outliers
            payoffs = train_df['sanrentan_payoff'].fillna(1000) # Default 10x
            w_odds = np.log1p(payoffs / 100)
            # Clip weight to avoid huge spikes (e.g. 500,000 payoff)
            w_odds = np.clip(w_odds, 0, 8) # log(3000) approx 8
            w_odds = w_odds / w_odds.mean()
            print(f"  Applied Phase 8 Odds-Weighted Learning. Mean weight factor: {w_odds.mean():.4f}")
            
        train_weights = w_time * w_odds
        print(f"  Final Weights: Min={train_weights.min():.4f}, Max={train_weights.max():.4f}, Mean={train_weights.mean():.4f}")

    # 1. LightGBM (LambdaRank)
    model_lgb = train_lgb(X_train, y_train_rank, X_val, y_val_rank, train_groups, val_groups, cat_features, train_weights=train_weights)
    lgb_val_scores = model_lgb.predict(X_val)
    # Calibrate: Ranking scores to pseudo-probabilities via Softmax per race
    lgb_val_preds = softmax_calibration(lgb_val_scores, val_groups)
    print(f"LGB (LambdaRank calibrated) LogLoss: {log_loss(y_val, lgb_val_preds):.4f}")
    
    # 2. CatBoost (Binary classification)
    model_cb = train_catboost(X_train, y_train, X_val, y_val, cat_features, train_weights=train_weights)
    cb_val_preds = model_cb.predict_proba(X_val)[:, 1]
    print(f"CB LogLoss: {log_loss(y_val, cb_val_preds):.4f}")
    
    # 3. Alternative Model
    model_alt, X_val_alt, alt_scaler = train_alt_model(X_train, y_train, X_val, y_val, train_weights=train_weights)
    alt_val_preds = model_alt.predict_proba(X_val_alt)[:, 1]
    print(f"Alt-Model LogLoss: {log_loss(y_val, alt_val_preds):.4f}")

    # 4. Factorization Machines (FM)
    fm_val_preds = None
    if ENABLE_FM_MODEL:
        model_fm, fm_val_preds, fm_le = train_fm(train_df, val_df)
        if not ENABLE_FM_MODEL: # Failed during training
            fm_val_preds = None

    # --- Stacking ---
    if ENABLE_ROI_OPTUNA_META:
        print(f"\n--- Meta-Stacking (V4 Final: Optuna ROI Optimization) ---")
        X_meta = np.vstack([lgb_val_preds, cb_val_preds, alt_val_preds]).T
        best_weights = get_optimal_weights_roi(X_meta, y_val, val_df)
        
        # Apply optimal weights
        final_preds = (best_weights['w_lgb'] * lgb_val_preds + 
                       best_weights['w_cb'] * cb_val_preds + 
                       best_weights['w_alt'] * alt_val_preds)
        
        # We don't use meta_model (LR) here, but for consistency we'll fit a dummy or just skip
        print(f"Stacked Model (V4 Final) LogLoss: {log_loss(y_val, final_preds):.4f}")
        evaluate_roi(val_df, final_preds, threshold=0.2)
    else:
        print(f"\n--- Meta-Stacking (V4: ENABLE_FM={ENABLE_FM_MODEL}) ---")
        if ENABLE_FM_MODEL and fm_val_preds is not None:
            X_meta = np.vstack([lgb_val_preds, cb_val_preds, alt_val_preds, fm_val_preds]).T
        else:
            X_meta = np.vstack([lgb_val_preds, cb_val_preds, alt_val_preds]).T
        
        meta_model = LogisticRegression()
        meta_model.fit(X_meta, y_val)
        
        final_preds = meta_model.predict_proba(X_meta)[:, 1]
        print(f"Stacked Model (V4) LogLoss: {log_loss(y_val, final_preds):.4f}")
        print(f"Stacked Model (V4) Accuracy: {accuracy_score(y_val, (final_preds > 0.5).astype(int)):.4f}")
        
        # ROI Evaluation
        evaluate_roi(val_df, final_preds)
    
    # Save Models
    if ENABLE_ROI_OPTUNA_META:
        with open(os.path.join(MODEL_DIR, 'stacking_weights.pkl'), 'wb') as f:
            pickle.dump(best_weights, f)
        print(f"  Saved Optuna-optimized weights to {os.path.join(MODEL_DIR, 'stacking_weights.pkl')}")
    else:
        with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'wb') as f:
            pickle.dump(meta_model, f)
            
    with open(os.path.join(MODEL_DIR, 'racer_mapping.pkl'), 'wb') as f:
        pickle.dump(racer_mapping, f)
    model_lgb.save_model(os.path.join(MODEL_DIR, 'lgb_model.txt'))
    model_cb.save_model(os.path.join(MODEL_DIR, 'cb_model.bin'))
    if 'model_alt' in locals():
        with open(os.path.join(MODEL_DIR, 'alt_model.pkl'), 'wb') as f:
            pickle.dump(model_alt, f)
        with open(os.path.join(MODEL_DIR, 'alt_scaler.pkl'), 'wb') as f:
            pickle.dump(alt_scaler, f)
    # Note: TabPFN itself is not easily pickled, will rely on retraining/inference if needed or fresh fit in App
    # In V3.1 we'll attempt to save TabPFN if possible, or handle it in App.
    
    # Visualization: Feature Importance (LGBM and CatBoost)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    lgb.plot_importance(model_lgb, ax=axes[0], title='LGBM Importance')
    cb_imp = pd.Series(model_cb.get_feature_importance(), index=features).sort_values()
    cb_imp.plot(kind='barh', ax=axes[1], title='CatBoost Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    print("\nFeature importance saved to feature_importance.png")

if __name__ == "__main__":
    main()
