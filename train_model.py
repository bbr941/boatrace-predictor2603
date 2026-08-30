import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import re
import os
import shutil
import argparse
from sklearn.metrics import ndcg_score

# Config
DATA_PATH = 'train_data_full.csv' if os.path.exists('train_data_full.csv') else 'boatrace_dataset_labeled_v2.csv'
MODEL_HONMEI = 'model_honmei.txt'
MODEL_HONMEI_BACKUP = 'model_honmei_backup.txt'
MODEL_ANA = 'model_ana.txt'

def preprocess_data(df):
    print("Preprocessing...")
    
    # Check for duplicate columns just in case
    df = df.loc[:, ~df.columns.duplicated()]

    # Advanced Feature Engineering (Before Type Conversion)
    df = add_advanced_features(df)
    
    # 1. Base Cleanup / Type Conversion
    for col in df.columns:
        if df[col].dtype == 'object':
            if col not in ['race_id', 'race_date', 'prior_results']:
                df[col] = df[col].astype('category')
    
    # Target: Relevance (Ranking)
    # 1st=10, 2nd=7, 3rd=4, others=0
    df['relevance'] = df['rank'].map({1: 10, 2: 7, 3: 4}).fillna(0)
    
    # Fill NaN
    if 'syn_win_rate' in df.columns:
        df['syn_win_rate'] = pd.to_numeric(df['syn_win_rate'], errors='coerce').fillna(0.0)

    # 欠損補完 (LightGBM ranker 互換性確保)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df

def add_advanced_features(df):
    print("  - Adding Advanced Features (Market Distortion & Performance Gaps)...")
    
    # 1. F (Flying) Analysis & ST Correction
    if 'prior_results' in df.columns:
        df['is_F_holder'] = df['prior_results'].astype(str).apply(lambda x: 1 if 'F' in x else 0)
    else:
        df['is_F_holder'] = 0
        
    st_col = 'course_avg_st' if 'course_avg_st' in df.columns else 'exhibition_start_timing' 
    if st_col in df.columns:
        df['corrected_st'] = df[st_col] + (df['is_F_holder'] * 0.05)
    else:
        df['corrected_st'] = 0.20
        
    df = df.sort_values(['race_id', 'boat_number'])
    
    prev_race_ids = df['race_id'].shift(1)
    prev_sts = df['corrected_st'].shift(1)
    
    df['inner_st_gap_corrected'] = df['corrected_st'] - prev_sts
    df.loc[df['race_id'] != prev_race_ids, 'inner_st_gap_corrected'] = 0.0
    
    # 2. Motor Evaluation Gap (Motor Rank - Tenji Rank)
    if 'motor_rate' in df.columns and 'exhibition_time' in df.columns:
        df['motor_rank'] = df.groupby('race_id')['motor_rate'].rank(ascending=False, method='min')
        df['tenji_rank'] = df.groupby('race_id')['exhibition_time'].rank(ascending=True, method='min')
        df['motor_gap'] = df['motor_rank'] - df['tenji_rank']
    else:
        df['motor_rank'] = 3.5
        df['tenji_rank'] = 3.5
        df['motor_gap'] = 0.0
    
    # 3. Specialist Gap
    if 'course_1st_rate' in df.columns and 'nat_win_rate' in df.columns:
        df['specialist_score'] = df['course_1st_rate'] - df['nat_win_rate']
    else:
        df['specialist_score'] = 0.0
        
    # 4. Winning Move Match
    def calc_rate(count_col, run_col):
        return df[count_col] / (df[run_col] + 1.0)
    
    if 'nige_count' in df.columns and 'course_run_count' in df.columns:
        df['my_nige_rate'] = calc_rate('nige_count', 'course_run_count')
        df['my_sashi_rate'] = calc_rate('sashi_count', 'course_run_count') if 'sashi_count' in df.columns else 0.0
        df['my_makuri_rate'] = calc_rate('makuri_count', 'course_run_count') if 'makuri_count' in df.columns else 0.0
        
        inner_nige_rate = df['my_nige_rate'].shift(1)
        df['sashi_potential'] = df['my_sashi_rate'] / (inner_nige_rate + 0.01)
        df.loc[df['boat_number'] == 1, 'sashi_potential'] = 0.0
        
        df['st_rank'] = df.groupby('race_id')['corrected_st'].rank(ascending=True)
        inner_st_rank = df['st_rank'].shift(1)
        df['makuri_potential'] = df['my_makuri_rate'] * inner_st_rank
        df.loc[df['boat_number'] == 1, 'makuri_potential'] = 0.0
    else:
        df['sashi_potential'] = 0.0
        df['makuri_potential'] = 0.0

    # 5. Venue Frame Bias
    bias_path = 'app_data/venue_frame_bias.csv'
    if os.path.exists(bias_path):
        bias_df = pd.read_csv(bias_path)
        bias_df['venue_code'] = bias_df['venue_code'].astype(str).str.zfill(2)
        bias_df['boat_number'] = bias_df['boat_number'].astype(int)
        
        venue_map = {
            '桐生': '01', '戸田': '02', '江戸川': '03', '平和島': '04', '多摩川': '05',
            '浜名湖': '06', '蒲郡': '07', '常滑': '08', '津': '09', '三国': '10',
            'びわこ': '11', '住之江': '12', '尼崎': '13', '鳴門': '14', '丸亀': '15',
            '児島': '16', '宮島': '17', '徳山': '18', '下関': '19', '若松': '20',
            '芦屋': '21', '福岡': '22', '唐津': '23', '大村': '24'
        }
        
        if 'venue_name' in df.columns:
            df['temp_venue_code'] = df['venue_name'].map(venue_map).fillna('00')
            df = df.merge(bias_df, left_on=['temp_venue_code', 'boat_number'], right_on=['venue_code', 'boat_number'], how='left')
            df.drop(columns=['temp_venue_code', 'venue_code'], inplace=True, errors='ignore')
            df['venue_frame_win_rate'] = df['venue_frame_win_rate'].fillna(df.groupby('boat_number')['venue_frame_win_rate'].transform('mean')).fillna(0.16)
        else:
            df['venue_frame_win_rate'] = 0.16
    else:
        df['venue_frame_win_rate'] = 0.16
        
    # 6. Rank-based features
    if 'racer_rank' in df.columns:
        rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
        df['rank_numeric'] = df['racer_rank'].map(rank_map).fillna(2).astype(int)
        
        if 'nat_win_rate' in df.columns:
            df['level_adjusted_win_rate'] = df['nat_win_rate'] * df['rank_numeric']
            rank_means = df.groupby('racer_rank')['nat_win_rate'].transform('mean')
            df['rank_skill_gap'] = (df['nat_win_rate'] - rank_means).fillna(0.0)
        else:
            df['level_adjusted_win_rate'] = 0.0
            df['rank_skill_gap'] = 0.0

    return df

def get_features(df, mode='honmei'):
    # Non-feature columns to exclude
    base_ignore = [
        'race_id', 'boat_number', 'racer_id', 'rank', 'relevance',
        'race_date', 'venue_name', 'racer_rank', 'prior_results',
        'weight_for_loss', 'pred_score', 'is_F_holder', 'temp_venue_code',
        'my_nige_rate', 'my_sashi_rate', 'my_makuri_rate', 'st_rank',
        'venue_code_x', 'venue_code_int', 'ana_relevance', 'weight_ana', 'proxy_odds',
        'weather', 'nige_count', 'makuri_count', 'makurizashi_count', 'sashi_count',
        'wintech_races_run', 'wintech_wins'
    ]
    
    odds_features = [
        'syn_win_rate', 'odds', 'prediction_odds', 'popularity', 
        'vote_count', 'win_share'
    ]
    
    all_cols = df.columns.tolist()
    candidates = [c for c in all_cols if c not in base_ignore]
    
    if mode == 'ana':
        final_feats = [c for c in candidates if not any(o in c for o in odds_features)]
        return final_feats
    else:
        # Honmei: Use all valid numeric and categorical features
        return candidates

    odds_features = [
        'syn_win_rate', 'odds', 'prediction_odds', 'popularity', 
        'vote_count', 'win_share' # Add any other odds-derived names
    ]
    
    all_cols = df.columns.tolist()
    candidates = [c for c in all_cols if c not in base_ignore]
    
    if mode == 'ana':
        # Remove odds features
        # Also check for partial matches if needed (e.g. 'odds' in name)
        final_feats = []
        for c in candidates:
            is_odds = False
            for o in odds_features:
                if o in c: # Simple substring check safe? e.g. "odds"
                    is_odds = True
                    break
            if not is_odds:
                final_feats.append(c)
        return final_feats
    else:
        # Honmei: Use all valid numeric and categorical features
        return candidates

def train_lgb_ranker(df, features, model_path, weight_col=None, label_col='relevance', split_date='2026-01-01'):
    print(f"\nTraining Model: {model_path} | Features: {len(features)}")

    
    # データをレース単位でソート
    df = df.sort_values(['race_date', 'race_id', 'boat_number']).reset_index(drop=True)
    
    # 時系列分割 (Out-of-Time: 2026年〜)
    unique_dates = sorted(df['race_date'].dropna().unique())
    if split_date is None or df['race_date'].max() < split_date or df['race_date'].min() >= split_date:
        split_idx = int(len(unique_dates) * 0.8)
        effective_split_date = unique_dates[split_idx]
    else:
        effective_split_date = split_date
        
    train_mask = df['race_date'] < effective_split_date
    test_mask = df['race_date'] >= effective_split_date
    
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    print(f"  実効データ分割基準日      : {effective_split_date} (Train: ~{effective_split_date}前日, Test: {effective_split_date}~)")
    print(f"  学習データ (Train) レコード数: {len(train_df):,} 行 ({train_df['race_id'].nunique():,} レース)")
    print(f"  検証データ (Test)  レコード数: {len(test_df):,} 行 ({test_df['race_id'].nunique():,} レース)")
    print("-" * 75, flush=True)
    
    # Groups (レースごとの出走頭数)
    train_grp = train_df.groupby('race_id', sort=False).size().to_numpy()
    test_grp = test_df.groupby('race_id', sort=False).size().to_numpy()
    
    # Weights
    w_train = None
    if weight_col and weight_col in train_df.columns:
        w_train = train_df[weight_col].to_numpy()
        w_train = np.maximum(w_train, 0.0)

    # Dataset
    dtrain = lgb.Dataset(train_df[features], label=train_df[label_col], group=train_grp, weight=w_train)
    dtest = lgb.Dataset(test_df[features], label=test_df[label_col], group=test_grp, reference=dtrain)
    
    # Params
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'ndcg_eval_at': [1, 2, 3],
        'learning_rate': 0.05,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 1,
        'verbose': -1,
        'random_state': 42,
        'n_jobs': -1
    }
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'test'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100)
        ]
    )
    
    if model_path:
        model.save_model(model_path)
        print(f"Model saved to: {model_path}")
    
    return model, train_df, test_df


def calc_dcg(relevances, k=3):
    rel = np.asarray(relevances)[:k]
    if len(rel) == 0:
        return 0.0
    gains = (2.0 ** rel) - 1.0
    discounts = np.log2(np.arange(len(rel)) + 2.0)
    return float(np.sum(gains / discounts))

def calc_ndcg(relevance_array, pred_scores, k=3):
    order = np.argsort(pred_scores)[::-1]
    pred_rel = np.asarray(relevance_array)[order]
    ideal_rel = np.sort(np.asarray(relevance_array))[::-1]
    
    dcg = calc_dcg(pred_rel, k)
    idcg = calc_dcg(ideal_rel, k)
    return (dcg / idcg) if idcg > 0 else 1.0

def evaluate_model_performance(model, test_df, features, label, old_model_path=None):
    print(f"\n" + "=" * 75)
    print(f"  📊 Out-of-Time 予測精度評価レポート: {label}")
    print("=" * 75)
    
    test_copy = test_df.copy()
    test_copy['pred_new'] = model.predict(test_copy[features])
    
    def calc_metrics(df_in, pred_col):
        # 1. Top-1 予想的中率 (最上位予測艇が1着になった割合)
        idx_top1 = df_in.groupby('race_id')[pred_col].idxmax()
        top1_acc = (df_in.loc[idx_top1, 'rank'] == 1).mean()
        
        # 2. 3連単 (Trifecta Top-1) 予想的中率
        def is_trifecta_match(g):
            if len(g) < 3: return False
            p_top3 = g.sort_values(pred_col, ascending=False)['boat_number'].values[:3]
            a_top3 = g.sort_values('rank', ascending=True)['boat_number'].values[:3]
            return np.array_equal(p_top3, a_top3)
        
        trifecta_acc = df_in.groupby('race_id').apply(is_trifecta_match, include_groups=False).mean()
        
        # 3. NDCG@1, @2, @3
        ndcg_list = {1: [], 2: [], 3: []}
        for _, g in df_in.groupby('race_id'):
            if len(g) < 3: continue
            y_true = g['relevance'].values
            y_score = g[pred_col].values
            if y_true.max() > 0:
                for k in [1, 2, 3]:
                    ndcg_list[k].append(calc_ndcg(y_true, y_score, k=k))
                    
        return {
            'top1_acc': top1_acc,
            'trifecta_acc': trifecta_acc,
            'ndcg1': np.mean(ndcg_list[1]),
            'ndcg2': np.mean(ndcg_list[2]),
            'ndcg3': np.mean(ndcg_list[3])
        }
    
    m_new = calc_metrics(test_copy, 'pred_new')

    
    m_old = None
    if old_model_path and os.path.exists(old_model_path):
        try:
            old_model = lgb.Booster(model_file=old_model_path)
            old_feats = old_model.feature_name()
            missing_old = [f for f in old_feats if f not in test_copy.columns]
            if not missing_old:
                test_copy['pred_old'] = old_model.predict(test_copy[old_feats])
                m_old = calc_metrics(test_copy, 'pred_old')
        except Exception as e:
            print(f"  (Note: 旧モデル比較スキップ: {e})")
            
    print(f"  指標 (Metric)             | 旧モデル (ベースライン) | 新モデル (再学習)   | 改善効果 (差分)")
    print(f"  --------------------------+-------------------------+---------------------+-----------------")
    
    if m_old:
        print(f"  NDCG@1                    | {m_old['ndcg1']:>23.5f} | {m_new['ndcg1']:>19.5f} | {m_new['ndcg1'] - m_old['ndcg1']:>+15.5f} {'(向上)' if m_new['ndcg1'] > m_old['ndcg1'] else ''}")
        print(f"  NDCG@2                    | {m_old['ndcg2']:>23.5f} | {m_new['ndcg2']:>19.5f} | {m_new['ndcg2'] - m_old['ndcg2']:>+15.5f} {'(向上)' if m_new['ndcg2'] > m_old['ndcg2'] else ''}")
        print(f"  NDCG@3                    | {m_old['ndcg3']:>23.5f} | {m_new['ndcg3']:>19.5f} | {m_new['ndcg3'] - m_old['ndcg3']:>+15.5f} {'(向上)' if m_new['ndcg3'] > m_old['ndcg3'] else ''}")
        print(f"  Top-1 的中率 (1着)        | {m_old['top1_acc']:>22.2%} | {m_new['top1_acc']:>18.2%} | {m_new['top1_acc'] - m_old['top1_acc']:>+14.2%} pt")
        print(f"  3連単 完全的中率          | {m_old['trifecta_acc']:>22.2%} | {m_new['trifecta_acc']:>18.2%} | {m_new['trifecta_acc'] - m_old['trifecta_acc']:>+14.2%} pt")
    else:
        print(f"  NDCG@1                    | {'-':>23} | {m_new['ndcg1']:>19.5f} | {'-':>15}")
        print(f"  NDCG@2                    | {'-':>23} | {m_new['ndcg2']:>19.5f} | {'-':>15}")
        print(f"  NDCG@3                    | {'-':>23} | {m_new['ndcg3']:>19.5f} | {'-':>15}")
        print(f"  Top-1 的中率 (1着)        | {'-':>23} | {m_new['top1_acc']:>18.2%} | {'-':>15}")
        print(f"  3連単 完全的中率          | {'-':>23} | {m_new['trifecta_acc']:>18.2%} | {'-':>15}")
    print("=" * 75 + "\n")
    
    return m_new, m_old

def print_feature_importance(model, features, top_n=25):
    gain = model.feature_importance(importance_type='gain')
    split = model.feature_importance(importance_type='split')
    total_gain = sum(gain)
    
    new_cross_features = [
        'wind_makuri_cross', 'strong_wind_makuri', 'wind_makurizashi_cross',
        'strong_wind_outer_adv', 'wind_nige_vulnerability',
        'wave_weight_prod', 'wave_weight_ratio', 'high_wave_heavy_penalty', 'high_wave_inner_risk',
        'ex_diff_from_race_min', 'ex_diff_from_race_mean', 'ex_rank_in_race',
        'ex_momentum_diff', 'ex_momentum_deviation', 'makurizashi_rate',
        'is_strong_wind', 'is_gale_wind', 'is_high_wave'
    ]
    
    feat_df = pd.DataFrame({
        'Feature': features,
        'Type': ['🌟 新規(クロス/モメンタム)' if f in new_cross_features else '従来(ベースライン)' for f in features],
        'Gain': gain,
        'Gain_Ratio (%)': (gain / total_gain) * 100.0,
        'Split_Count': split
    }).sort_values(by='Gain', ascending=False).reset_index(drop=True)
    
    print("=" * 75)
    print(f"  🏆 Feature Importance ランキング (Gain 寄与度順 Top {top_n})")
    print("=" * 75)
    print(f"  Rank | Feature Name                 | Category                | Gain Ratio | Split Count")
    print(f"  -----+------------------------------+-------------------------+------------+------------")
    for i, row in feat_df.head(top_n).iterrows():
        is_new_mark = "🌟" if "新規" in str(row['Type']) else "  "
        print(f"  {i+1:>4d} | {row['Feature']:<28} | {row['Type']:<23} | {row['Gain_Ratio (%)']:>9.2f}% | {row['Split_Count']:>10d} {is_new_mark}")
    print("-" * 75)
    
    new_gain_sum = feat_df[feat_df['Type'].str.contains('新規')]['Gain_Ratio (%)'].sum()
    print(f"  🌟 新規環境クロス・モメンタム特徴量の総合寄与度: {new_gain_sum:.2f}%")
    print("=" * 75 + "\n")
    return feat_df

def main():
    parser = argparse.ArgumentParser(description="Train Boatrace LightGBM Models")
    parser.add_argument('--data', type=str, default=DATA_PATH, help="Path to CSV dataset")
    parser.add_argument('--split_date', type=str, default='2026-01-01', help="Out-of-Time split date")
    parser.add_argument('--save_model', action='store_true', default=True, help="Save trained model to model_honmei.txt")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data):
        print(f"Data not found: {args.data}")
        return
    
    print(f"Loading dataset from: {args.data} ...")
    df = pd.read_csv(args.data)
    df = preprocess_data(df)
    
    # --- Model A: Honmei (Accuracy - LambdaRank) ---
    feats_honmei = get_features(df, mode='honmei')
    print(f"\nHonmei Model Features ({len(feats_honmei)}): {feats_honmei}")
    
    # バックアップの作成
    if os.path.exists(MODEL_HONMEI):
        shutil.copyfile(MODEL_HONMEI, MODEL_HONMEI_BACKUP)
        print(f"Existing model backed up to: {MODEL_HONMEI_BACKUP}")
    
    model_h, train_df, test_df = train_lgb_ranker(df, feats_honmei, MODEL_HONMEI, weight_col=None, split_date=args.split_date)
    
    # 精度評価
    evaluate_model_performance(model_h, test_df, feats_honmei, "Honmei (LambdaRank)", old_model_path=MODEL_HONMEI_BACKUP)
    
    # Feature Importance
    print_feature_importance(model_h, feats_honmei)
    
    # モデル保存
    if args.save_model:
        model_h.save_model(MODEL_HONMEI)
        print(f"✅ Saved updated model to: {MODEL_HONMEI}")
    
    print("\nAll Training and Evaluation Completed.")

if __name__ == "__main__":
    main()

