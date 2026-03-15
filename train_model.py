import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import re
import os

# Config
DATA_PATH = 'boatrace_dataset_labeled_v2.csv'
MODEL_HONMEI = 'model_honmei.txt'
MODEL_ANA = 'model_ana.txt'

def preprocess_data(df):
    print("Preprocessing...")
    
    # Check for duplicate columns just in case
    df = df.loc[:, ~df.columns.duplicated()]

    # Advanced Feature Engineering (Before Type Conversion)
    df = add_advanced_features(df)
    
    # 1. Base Cleanup / Type Conversion
    # Convert object columns to category for LGBM
    # Exclude ID/Date columns from Feature consideration automatically later,
    # but valid "text" features might be category.
    
    # Explicitly ignore non-feature columns for categorical conversion check if needed,
    # but generally safe to convert all objects.
    
    # However, 'race_date' might be object.
    # 'race_id', 'boat_number', 'racer_id', 'rank' are numeric/IDs.
    
    # Define potential feature candidates (all columns)
    # We will filter them later per model.
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if it looks categorical or just ID
            # 'race_id' is unique per race, not useful as category usually (too high cardinality).
            if col not in ['race_id', 'race_date', 'prior_results']:
                df[col] = df[col].astype('category')
    
    # Target: Relevance (Ranking)
    # 1st=10, 2nd=7, 3rd=4, others=0
    df['relevance'] = df['rank'].map({1: 10, 2: 7, 3: 4}).fillna(0)
    
    # Fill NaN
    # LightGBM handles NaN, but syn_win_rate NaN should probably be 0
    if 'syn_win_rate' in df.columns:
        df['syn_win_rate'] = df['syn_win_rate'].fillna(0)

    return df

def add_advanced_features(df):
    print("  - Adding Advanced Features (Market Distortion)...")
    
    # 1. F (Flying) Analysis & ST Correction
    # Parse 'prior_results' like "[4 5 F 1]"
    # Regex look for 'F'
    if 'prior_results' in df.columns:
        df['is_F_holder'] = df['prior_results'].astype(str).apply(lambda x: 1 if 'F' in x else 0)
    else:
        df['is_F_holder'] = 0
        
    # Corrected ST (Penalty +0.05 if F holder)
    # Use 'course_avg_st' if available, otherwise 'avg_st' or similar. 
    # Validating col availability:
    st_col = 'course_avg_st' if 'course_avg_st' in df.columns else 'exhibition_start_timing' 
    # Note: Training data might use 'avg_start_timing' or 'course_avg_st'.
    
    if st_col in df.columns:
        df['corrected_st'] = df[st_col] + (df['is_F_holder'] * 0.05)
    else:
        df['corrected_st'] = 0.20 # Default
        
    # Update/Create 'inner_st_gap' using corrected_st
    # Race-wise operation. Requires GroupBy or sorting.
    # To be efficient, we can do: sorted by race_id, boat_number.
    # Then shift.
    # Ensure sorted
    df = df.sort_values(['race_id', 'boat_number'])
    
    # Inner ST Gap: Me - Inner Boat. (Positive = I am Slower)
    # GroupBy shift is slow. Fast vector approach:
    # Shifted ST
    prev_race_ids = df['race_id'].shift(1)
    prev_sts = df['corrected_st'].shift(1)
    
    # Where race_id matches, gap = my_st - prev_st. Boat 1 has 0 gap.
    df['inner_st_gap_corrected'] = df['corrected_st'] - prev_sts
    # Mask where race_id changed (Boat 1)
    df.loc[df['race_id'] != prev_race_ids, 'inner_st_gap_corrected'] = 0.0
    
    # 2. Motor Evaluation Gap (Motor Rank - Tenji Rank)
    # Lower Rank is Better (1st).
    # Motor Rate: Higher is Better -> Rank Descending
    # Ex Time: Lower is Better -> Rank Ascending
    
    # GroupBy Rank
    # Use 'transform' for speed
    df['motor_rank'] = df.groupby('race_id')['motor_rate'].rank(ascending=False, method='min')
    df['tenji_rank'] = df.groupby('race_id')['exhibition_time'].rank(ascending=True, method='min')
    df['motor_gap'] = df['motor_rank'] - df['tenji_rank']
    # Interpretation: MotorRank(6=Bad) - TenjiRank(1=Good) = +5 (Good Gap: Numbers bad but running well)
    
    # 3. Specialist Gap
    # course_1st_rate - nat_win_rate
    # Check cols
    if 'course_1st_rate' in df.columns and 'nat_win_rate' in df.columns:
        df['specialist_score'] = df['course_1st_rate'] - df['nat_win_rate']
    else:
        df['specialist_score'] = 0.0
        
    # 4. Winning Move Match
    # 2-Course Sashi Potential: (2c Sashi% / 1c Nige%)
    # 3-Course Makuri Potential: (3c Makuri% * 2c AvgST Rank)
    
    # We need access to neighbor's stats.
    # Using shift again.
    
    # 1-Course Nige Rate (from Boat 1)
    # We need to broadcast Boat 1's Nige Rate to Boat 2.
    # Actually, simpler: Nige Rate of inner boat.
    
    # Shifted Nige (Inner Boat Nige)
    inner_nige = df['nige_count'].shift(1) # Assuming nige_count is rate or count? Prompt said "count/run".
    # We likely have rates or counts. Let's assume rate if normalized, or just use raw count if consistent.
    # 'nige_count' in existing feature engineering is raw count. 'course_run_count' is denominator.
    
    # Rate calculation helper
    def calc_rate(count_col, run_col):
        return df[count_col] / (df[run_col] + 1.0) # avoid 0 div
    
    if 'nige_count' in df.columns and 'course_run_count' in df.columns:
        df['my_nige_rate'] = calc_rate('nige_count', 'course_run_count')
        df['my_sashi_rate'] = calc_rate('sashi_count', 'course_run_count')
        df['my_makuri_rate'] = calc_rate('makuri_count', 'course_run_count')
        
        # Shift rates
        inner_nige_rate = df['my_nige_rate'].shift(1)
        
        # 2-Course Sashi Potential
        # Sashi Potential = My Sashi Rate / Inner Nige Rate (Small inner nige -> High Sashi opp)
        # Avoid 0 div
        df['sashi_potential'] = df['my_sashi_rate'] / (inner_nige_rate + 0.01)
        # Only valid for Boat 2? Or generally "Sashi against inner"?
        # Prompt said "2-Course Sashi Expectation". So valid when Boat=2.
        # But we can generalize to "Sashi vs Inner".
        df.loc[df['boat_number'] == 1, 'sashi_potential'] = 0
        
        # 3-Course Makuri Potential
        # Need Inner Boat's ST Rank.
        # ST Rank in Race.
        df['st_rank'] = df.groupby('race_id')['corrected_st'].rank(ascending=True)
        inner_st_rank = df['st_rank'].shift(1)
        
        # Makuri Potential = My Makuri Rate * Inner ST Rank (Larger Rank=Slower=Good for Makuri)
        df['makuri_potential'] = df['my_makuri_rate'] * inner_st_rank
        df.loc[df['boat_number'] == 1, 'makuri_potential'] = 0
        
        # Cleanup temp cols if necessary, or keep as features
        
    else:
        df['sashi_potential'] = 0.0
        df['makuri_potential'] = 0.0

    # 5. Venue Frame Bias (from Deme_Ranking)
    # Load bias table
    bias_path = 'app_data/venue_frame_bias.csv'
    if os.path.exists(bias_path):
        bias_df = pd.read_csv(bias_path)
        # Ensure Types
        bias_df['venue_code'] = bias_df['venue_code'].astype(str).str.zfill(2)
        bias_df['boat_number'] = bias_df['boat_number'].astype(int)
        
        # Prepare DF venue_code
        # Assuming venue_name exists. Map Name -> Code.
        # Standard 24 Venue Map
        venue_map = {
            '桐生': '01', '戸田': '02', '江戸川': '03', '平和島': '04', '多摩川': '05',
            '浜名湖': '06', '蒲郡': '07', '常滑': '08', '津': '09', '三国': '10',
            'びわこ': '11', '住之江': '12', '尼崎': '13', '鳴門': '14', '丸亀': '15',
            '児島': '16', '宮島': '17', '徳山': '18', '下関': '19', '若松': '20',
            '芦屋': '21', '福岡': '22', '唐津': '23', '大村': '24'
        }
        
        if 'venue_name' in df.columns:
            # Create temp join col
            df['temp_venue_code'] = df['venue_name'].map(venue_map).fillna('00')
            
            # Merge
            df = df.merge(bias_df, left_on=['temp_venue_code', 'boat_number'], right_on=['venue_code', 'boat_number'], how='left')
            
            # Drop temp/redundant
            df.drop(columns=['temp_venue_code', 'venue_code'], inplace=True, errors='ignore')
            
            # Fill NaNs (some venues might be missing?)
            df['venue_frame_win_rate'] = df['venue_frame_win_rate'].fillna(df.groupby('boat_number')['venue_frame_win_rate'].transform('mean'))
            df['venue_frame_win_rate'] = df['venue_frame_win_rate'].fillna(0.16) # Fallback 1/6
            
        else:
            df['venue_frame_win_rate'] = 0.0
    else:
        df['venue_frame_win_rate'] = 0.0

    return df

def get_features(df, mode='honmei'):
    # Common ignore list
    # Ensure new features are NOT here.
    base_ignore = [
        'race_id', 'boat_number', 'racer_id', 'rank', 'relevance',
        'race_date', # Date usually not a direct feature unless processed
        'venue_name', # captured by venue_code or category
        'prior_results', # Raw string
        'weight_for_loss', # Internal column
        'pred_score', # artifact
        
        # Intermediate / Redundant
        'is_F_holder', 'temp_venue_code',
        'my_nige_rate', 'my_sashi_rate', 'my_makuri_rate', 'st_rank',

        # Low Importance / Noisy Features (Step 1 Optimization)
        'tenji_rank', 'is_linear_leader', 'high_wind_alert', 
        'inner_st_gap', 'course_avg_st', 'exhibition_start_timing',
        'wave_height', 'wind_direction', 'wind_vector_lat', 'wind_vector_long',
        'makuri_count', 'sashi_count', 
        'venue_course_2nd_rate', 'venue_course_3rd_rate',
        'boat_rate', # Low importance (4507) vs Motor (higher usually)
        'venue_code_x' # ID-like
    ]
    
    # Features derived from odds (Forbidden in Ana)
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
        # Honmei: Use everything including odds
        return candidates

def train_lgb_ranker(df, features, model_path, weight_col=None, label_col='relevance'):
    print(f"\nTraining Model: {model_path} | Features: {len(features)}")
    
    # Split
    unique_races = df['race_id'].unique()
    split_idx = int(len(unique_races) * 0.8)
    train_races = unique_races[:split_idx]
    test_races = unique_races[split_idx:]
    
    train_df = df[df['race_id'].isin(train_races)]
    test_df = df[df['race_id'].isin(test_races)]
    
    # Groups
    train_grp = train_df.groupby('race_id').size().to_numpy()
    test_grp = test_df.groupby('race_id').size().to_numpy()
    
    # Weights
    w_train = None
    if weight_col:
        w_train = train_df[weight_col].to_numpy()
        # Ensure no negative weights
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
        'verbose': -1,
        'random_state': 42
    }
    
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dtest],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(period=100)
        ]
    )
    
    model.save_model(model_path)
    return model, test_df

def evaluate_trifecta(model, df, features, label):
    print(f"Evaluating {label}...")
    df['pred_score'] = model.predict(df[features])
    
    # 3連単 (Trifecta) Evaluation
    predictions = df.groupby('race_id').apply(
        lambda x: x.sort_values('pred_score', ascending=False)['boat_number'].tolist()
    )
    actuals = df.groupby('race_id').apply(
        lambda x: x.sort_values('rank', ascending=True)['boat_number'].tolist()
    )
    
    total = 0
    exact3 = 0
    
    # Use index intersection to match races
    common_idx = predictions.index.intersection(actuals.index)
    
    for rid in common_idx:
        p = predictions[rid][:3]
        a = actuals[rid][:3]
        if len(p) == 3 and len(a) == 3:
            total += 1
            if p == a:
                exact3 += 1
                
    if total > 0:
        print(f"[{label}] Trifecta Accuracy: {exact3/total:.2%} ({exact3}/{total})")
    else:
        print(f"[{label}] No valid races for evaluation.")

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data not found: {DATA_PATH}")
        return
    
    df = pd.read_csv(DATA_PATH)
    df = preprocess_data(df)
    
    # --- Model A: Honmei (Accuracy) ---
    feats_honmei = get_features(df, mode='honmei')
    model_h, test_h = train_lgb_ranker(df, feats_honmei, MODEL_HONMEI, weight_col=None)
    evaluate_trifecta(model_h, test_h, feats_honmei, "Honmei")
    
    # --- Model B: Ana (Recovery/Payout - Special Tuning) ---
    print("\n--- Configuring Ana Model (High Dividend Special) ---")
    
    # 1. Calculate Proxy Odds (1 / syn_win_rate)
    if 'syn_win_rate' in df.columns:
        # Avoid division by zero
        df['proxy_odds'] = df['syn_win_rate'].apply(lambda x: 1.0/x if x > 0.001 else 1.0)
    else:
        df['proxy_odds'] = 1.0 # Fallback
        
    # 2. Define Custom Relevance for Ana
    # Base: Copy relevance
    df['ana_relevance'] = df['relevance'].copy()
    
    # Mask: Winning Boat (Rank 1)
    mask_win = (df['rank'] == 1)
    
    # Strategy 1: "Cheap Win" -> 0 (Ignore)
    # Condition: Rank 1 AND Odds < 10.0
    mask_cheap = mask_win & (df['proxy_odds'] < 10.0)
    df.loc[mask_cheap, 'ana_relevance'] = 0
    print(f"  - Cheap Wins (Odds<10) masked to 0: {mask_cheap.sum()} rows")
    
    # Strategy 2: "High Dividend Win" -> Boost Relevance
    # Condition: Rank 1 AND Odds >= 10.0
    # Relevance = Odds (Direct usage)
    mask_high = mask_win & (df['proxy_odds'] >= 10.0)
    df.loc[mask_high, 'ana_relevance'] = df.loc[mask_high, 'proxy_odds']
    print(f"  - High Dividend Wins (Odds>=10) set to Odds: {mask_high.sum()} rows")
    
    # Cast to int for LightGBM Ranking and Clip to valid range (0-30)
    # LightGBM default label range is limited.
    df['ana_relevance'] = df['ana_relevance'].astype(int).clip(upper=30)
    
    # 3. Define Weighting
    # Strategy: Weight = Odds (Direct or Squared)
    # User requested: "tansho_odds itself or squared"
    # We use Odds directly for now.
    df['weight_ana'] = 1.0 # Default
    
    # Apply to all Rank 1 rows? Or just High Dividend?
    # User said "weight = tansho_odds". usually applied to the positive samples.
    # We will apply weight=Odds to ALL Valid Wins (High Dividend only, since Low are 0 relevance)
    # Actually, if Relevance is 0, Weight doesn't matter much for ranking (it pushes it down? No, it just ignores it).
    # But for "High Dividend", we want to emphasize it.
    df.loc[mask_high, 'weight_ana'] = df.loc[mask_high, 'proxy_odds']
    
    # Also boost weights for Rank 2, 3? Maybe not. Keep 1.0.
    
    feats_ana = get_features(df, mode='ana')
    print(f"  - Ana Features: {len(feats_ana)} features (Odds excluded)")
    
    model_a, test_a = train_lgb_ranker(df, feats_ana, MODEL_ANA, weight_col='weight_ana', label_col='ana_relevance')
    evaluate_trifecta(model_a, test_a, feats_ana, "Ana")

    print("\nAll Done.")

if __name__ == "__main__":
    main()
