import os
import sys
import datetime
import itertools
import asyncio
import logging
import traceback
import time
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

# --- Path Adjustment (Must be at the very top) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from data_fetcher import get_realtime_data
import config

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
st.set_page_config(page_title="BoatRace AI V3.1+ - Plan B Strategy", layout="wide")

MODEL_HONMEI_PATH = os.path.join(ROOT_DIR, 'model_honmei.txt')
MODEL_ANA_PATH = os.path.join(ROOT_DIR, 'model_ana.txt')
DATA_DIR = os.path.join(ROOT_DIR, 'app_data')

# Windows用非同期パッチ
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

@st.cache_resource
def load_models():
    """学習済み本命・穴モデルをロードする（キャッシュ化）"""
    if not os.path.exists(MODEL_HONMEI_PATH) or not os.path.exists(MODEL_ANA_PATH):
        st.error(f"Models not found in ROOT: {ROOT_DIR}")
        return None
    try:
        model_h = lgb.Booster(model_file=MODEL_HONMEI_PATH)
        model_a = lgb.Booster(model_file=MODEL_ANA_PATH)
        st.sidebar.success("Honmei & Ana Models Loaded Successfully!")
        return {'honmei': model_h, 'ana': model_a}
    except Exception as e:
        st.error(f"FATAL Model Load Error: {e}")
        st.code(traceback.format_exc())
        return None

def add_advanced_features(df):
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
        
    # Inner ST Gap Corrected
    df = df.sort_values(['race_id', 'boat_number'])
    prev_sts = df['corrected_st'].shift(1)
    df['inner_st_gap_corrected'] = df['corrected_st'] - prev_sts
    df.loc[df['boat_number'] == 1, 'inner_st_gap_corrected'] = 0.0
    
    # 2. Motor Gap
    if 'motor_rate' in df.columns and 'exhibition_time' in df.columns:
        df['motor_rank'] = df.groupby('race_id')['motor_rate'].rank(ascending=False, method='min')
        df['tenji_rank'] = df.groupby('race_id')['exhibition_time'].rank(ascending=True, method='min')
        df['motor_gap'] = df['motor_rank'] - df['tenji_rank']
    else:
        df['motor_gap'] = 0.0
        
    # 3. Specialist Gap
    if 'venue_course_1st_rate' in df.columns and 'nat_win_rate' in df.columns:
        df['specialist_score'] = df['venue_course_1st_rate'] - df['nat_win_rate']
    else:
        df['specialist_score'] = 0.0
        
    # 4. Winning Move Match
    if 'nige_count' in df.columns and 'course_run_count' in df.columns:
        df['my_nige_rate'] = df['nige_count'] / (df['course_run_count'] + 1.0)
        df['my_sashi_rate'] = df['sashi_count'] / (df['course_run_count'] + 1.0)
        df['my_makuri_rate'] = df['makuri_count'] / (df['course_run_count'] + 1.0)
        
        inner_nige_rate = df['my_nige_rate'].shift(1)
        df['sashi_potential'] = df['my_sashi_rate'] / (inner_nige_rate + 0.01)
        df.loc[df['boat_number'] == 1, 'sashi_potential'] = 0
        
        df['st_rank'] = df.groupby('race_id')['corrected_st'].rank(ascending=True)
        inner_st_rank = df['st_rank'].shift(1)
        df['makuri_potential'] = df['my_makuri_rate'] * inner_st_rank
        df.loc[df['boat_number'] == 1, 'makuri_potential'] = 0
    else:
        df['sashi_potential'] = 0.0
        df['makuri_potential'] = 0.0
        
    # 5. Venue Frame Bias
    bias_path = os.path.join(DATA_DIR, 'venue_frame_bias.csv')
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
            
            if 'venue_frame_win_rate' in df.columns:
                df['venue_frame_win_rate'] = df['venue_frame_win_rate'].fillna(0.16)
            else:
                 df['venue_frame_win_rate'] = 0.0
        else:
            df['venue_frame_win_rate'] = 0.0
    else:
        df['venue_frame_win_rate'] = 0.0
        
    return df

class FeatureEngineer:
    @staticmethod
    def process(df, venue_name, debug_mode=False):
        df['venue_name'] = venue_name
        
        try:
            r_course = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_course.csv'))
            r_venue = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_venue.csv'))
            v_course = pd.read_csv(os.path.join(DATA_DIR, 'static_venue_course.csv'))
            r_params = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_params.csv'))
            
            df['racer_id'] = df['racer_id'].astype(int)
            df['pred_course'] = df['pred_course'].astype(int)
            r_course['RacerID'] = r_course['RacerID'].astype(int)
            r_course['Course'] = r_course['Course'].astype(int)
            r_venue['RacerID'] = r_venue['RacerID'].astype(int)
            v_course['course_number'] = v_course['course_number'].astype(int)
            r_params['racer_id'] = r_params['racer_id'].astype(int)

            df = df.merge(r_course, left_on=['racer_id', 'pred_course'], right_on=['RacerID', 'Course'], how='left')
            df.rename(columns={
                'RacesRun': 'course_run_count',
                'QuinellaRate': 'course_quinella_rate',
                'TrifectaRate': 'course_trifecta_rate',
                'FirstPlaceRate': 'course_1st_rate',
                'AvgStartTiming': 'course_avg_st',
                'Nige': 'nige_count', 
                'Makuri': 'makuri_count',
                'Sashi': 'sashi_count'
            }, inplace=True)

            venue_map_rev = {
                '桐生': 1, '戸田': 2, '江戸川': 3, '平和島': 4, '多摩川': 5,
                '浜名湖': 6, '蒲郡': 7, '常滑': 8, '津': 9, '三国': 10,
                'びわこ': 11, '住之江': 12, '尼崎': 13, '鳴門': 14, '丸亀': 15,
                '児島': 16, '宮島': 17, '徳山': 18, '下関': 19, '若松': 20,
                '芦屋': 21, '福岡': 22, '唐津': 23, '大村': 24
            }
            df['venue_code_int'] = df['venue_name'].map(venue_map_rev).fillna(0).astype(int)
            r_venue['Venue'] = pd.to_numeric(r_venue['Venue'], errors='coerce').fillna(0).astype(int)
            
            df = df.merge(r_venue, left_on=['racer_id', 'venue_code_int'], right_on=['RacerID', 'Venue'], how='left')
            
            if 'local_win_rate' in df.columns:
                 df['local_win_rate'] = df['local_win_rate'].replace(0.0, np.nan)
                 if 'WinRate' in df.columns:
                     df['local_win_rate'] = df['local_win_rate'].fillna(df['WinRate'])
            elif 'WinRate' in df.columns:
                 df['local_win_rate'] = df['WinRate']

            df = df.merge(v_course, left_on=['venue_name', 'pred_course'], right_on=['venue_name', 'course_number'], how='left')
            df.rename(columns={'rate_1st': 'venue_course_1st_rate', 'rate_2nd': 'venue_course_2nd_rate', 'rate_3rd': 'venue_course_3rd_rate'}, inplace=True)

            df = df.merge(r_params, on='racer_id', how='left')
            
        except Exception: pass
        
        required_cols = ['makuri_count', 'nige_count', 'sashi_count', 'nat_win_rate', 'course_run_count', 'local_win_rate', 'course_quinella_rate']
        for c in required_cols:
            if c not in df.columns: df[c] = 0.0
            
        def parse_prior(x):
            if isinstance(x, (int, float)): return float(x)
            if not isinstance(x, str): return 3.5
            try:
                x_c = re.sub(r'[欠失FLS]', '', x)
                parts = x_c.split()
                ranks = [float(p) for p in parts if p.isdigit()]
                if ranks: return sum(ranks)/len(ranks)
            except: pass
            return 3.5
            
        df['series_avg_rank'] = df['prior_results'].apply(parse_prior)
        df['makuri_rate'] = df['makuri_count'] / df['course_run_count'].replace(0, 1)
        df['nige_rate'] = df['nige_count'] / df['course_run_count'].replace(0, 1)

        # Advanced Features
        df = add_advanced_features(df)

        df = df.sort_values('pred_course')
        st_col = 'corrected_st' if 'corrected_st' in df.columns else 'exhibition_start_timing'
        
        df['inner_st'] = df[st_col].shift(1).fillna(0)
        df['inner_st_gap'] = df[st_col] - df['inner_st']
        df['outer_st'] = df[st_col].shift(-1).fillna(0)
        avg_neighbor = (df['inner_st'] + df['outer_st']) / 2
        df['slit_formation'] = df[st_col] - avg_neighbor

        c1_nige = df.loc[df['pred_course']==1, 'nige_rate']
        val = c1_nige.values[0] if len(c1_nige) > 0 else 0.5
        df['anti_nige_potential'] = df['makuri_rate'] * (1 - val)
        
        df['wall_strength'] = df['course_quinella_rate'].shift(1).fillna(0)
        df['follow_potential'] = df['makuri_rate'].shift(1).fillna(0) * df['course_quinella_rate']
        
        mean_t = df['exhibition_time'].mean()
        std_t = df['exhibition_time'].std()
        if std_t == 0: std_t = 1
        df['tenji_z_score'] = (mean_t - df['exhibition_time']) / std_t
        df['linear_rank'] = df['exhibition_time'].rank(method='min', ascending=True)
        df['is_linear_leader'] = (df['linear_rank'] == 1).astype(int)
        
        if 'weight_x' in df.columns: df['weight'] = df['weight_x']
        if 'weight' not in df.columns: df['weight'] = 52.0
        df['weight_diff'] = df['weight'] - df['weight'].mean()
        df['high_wind_alert'] = (df['wind_speed'] >= 5).astype(int)
        
        df['nat_win_rate'] = pd.to_numeric(df['nat_win_rate'], errors='coerce').fillna(0.0)
        df['local_win_rate'] = pd.to_numeric(df['local_win_rate'], errors='coerce').fillna(0.0)
        df['local_perf_diff'] = df['local_win_rate'] - df['nat_win_rate']

        # Wind Vector
        def wind_deg_from_int(x): return (x - 1) * 22.5 if 1 <= x <= 16 else 0
        df['wind_angle_deg'] = df['wind_direction'].apply(wind_deg_from_int)
        venue_tailwind_from = {
            '桐生': 135, '戸田': 90, '江戸川': 180, '平和島': 180, '多摩川': 270,
            '浜名湖': 180, '蒲郡': 270, '常滑': 270, '津': 135, '三国': 180,
            'びわこ': 225, '住之江': 270, '尼崎': 90, '鳴門': 135, '丸亀': 15,
            '児島': 225, '宮島': 270, '徳山': 135, '下関': 270, '若松': 270,
            '芦屋': 135, '福岡': 0, '唐津': 135, '大村': 315
        }
        df['venue_tailwind_deg'] = df['venue_name'].map(venue_tailwind_from).fillna(0)
        angle_diff_rad = np.radians(df['wind_angle_deg'] - df['venue_tailwind_deg'])
        df['wind_vector_long'] = df['wind_speed'] * np.cos(angle_diff_rad)
        df['wind_vector_lat'] = df['wind_speed'] * np.sin(angle_diff_rad)

        if 'race_date' not in df.columns: df['race_date'] = '20000101'
        
        # Win Direction Mapping
        wind_map = {
            1: '北', 2: '北北東', 3: '北東', 4: '東北東', 5: '東', 6: '東南東', 7: '南東', 8: '南南東',
            9: '南', 10: '南南西', 11: '南西', 12: '西南西', 13: '西', 14: '西北西', 15: '北西', 16: '北北西'
        }
        if pd.api.types.is_numeric_dtype(df['wind_direction']):
             df['wind_direction'] = df['wind_direction'].map(wind_map).fillna(df['wind_direction'])
             df['wind_direction'] = df['wind_direction'].astype(str)
             df['wind_direction'] = df['wind_direction'].replace('nan', '')

        for col in df.columns:
            if col not in ['race_id', 'race_date', 'venue_name', 'prior_results', 'wind_direction', 'branch', 'class', 'racer_class']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        ignore_cols = ['race_id', 'race_date', 'prior_results', 'pred_score', 'weight_for_loss', 'relevance', 'rank']
        for col in df.columns:
            if col in ignore_cols: continue
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')

        return df

def calculate_plackett_luce_probs(honmei_scores_dict):
    boats = list(honmei_scores_dict.keys())
    scores = np.array([honmei_scores_dict[b] for b in boats])
    max_score = np.max(scores)
    exp_scores = np.exp(scores - max_score)
    p1 = exp_scores / np.sum(exp_scores)
    p1_dict = {boats[i]: p1[i] for i in range(len(boats))}
    
    combos = list(itertools.permutations(boats, 3))
    pl_probs = []
    for c in combos:
        b1, b2, b3 = c
        prob1 = p1_dict[b1]
        denom2 = 1.0 - p1_dict[b1]
        if denom2 <= 0: denom2 = 1e-9
        prob2 = p1_dict[b2] / denom2
        denom3 = 1.0 - p1_dict[b1] - p1_dict[b2]
        if denom3 <= 0: denom3 = 1e-9
        prob3 = p1_dict[b3] / denom3
        total_prob = prob1 * prob2 * prob3
        combo_str = f"{b1}-{b2}-{b3}"
        pl_probs.append({'combo': combo_str, 'prob': total_prob})
        
    pl_probs.sort(key=lambda x: x['prob'], reverse=True)
    
    p1_sorted = sorted(p1_dict.items(), key=lambda x: x[1], reverse=True)
    max_p1 = float(p1_sorted[0][1])
    prob_gap = float(pl_probs[0]['prob'] - pl_probs[1]['prob']) if len(pl_probs) >= 2 else 0.0
    
    return pl_probs, max_p1, prob_gap

def select_hybrid_formation_plan_b(pl_probs, ana_scores_dict, all_odds):
    if not pl_probs or not all_odds: return []
    top_combo = pl_probs[0]['combo']
    top_odds = all_odds.get(top_combo, 0)
    if top_odds < 1: return []
    
    N = int(min(8, math.floor(top_odds)))
    if N < 1: return []
    
    selected = [p['combo'] for p in pl_probs[:N]]
    best_ana_boat = max(ana_scores_dict, key=ana_scores_dict.get)
    best_ana_boat_str = str(best_ana_boat)
    
    ana_in_3rd = any(c.split('-')[2] == best_ana_boat_str for c in selected)
    if not ana_in_3rd:
        parts = top_combo.split('-')
        b1, b2 = parts[0], parts[1]
        if best_ana_boat_str != b1 and best_ana_boat_str != b2:
            new_combo = f"{b1}-{b2}-{best_ana_boat_str}"
            if new_combo not in selected:
                selected[-1] = new_combo
    return selected

def calculate_funds_distribution(selected_combos, pl_probs_list, all_odds, base_return, bonus_budget):
    if not selected_combos: return {}
    pl_probs_dict = {p['combo']: p['prob'] for p in pl_probs_list}
    for c in selected_combos:
        if all_odds.get(c, 0) < 1.01: return {}
    
    sum_p = sum(pl_probs_dict.get(c, 0) for c in selected_combos)
    if sum_p <= 0: sum_p = 1.0
    
    bets = {}
    for c in selected_combos:
        o = all_odds[c]
        p = pl_probs_dict.get(c, 0)
        b_base = base_return / o
        b_bonus = bonus_budget * (p / sum_p)
        raw_bet = b_base + b_bonus
        bet_100 = max(100, round(raw_bet / 100) * 100)
        bets[c] = bet_100
        
    total_bet = sum(bets.values())
    for c, b in bets.items():
        if b * all_odds[c] <= total_bet:
            bets_flat = {c: 100 for c in selected_combos}
            total_flat = len(selected_combos) * 100
            for cf, bf in bets_flat.items():
                if bf * all_odds[cf] <= total_flat: return {}
            return bets_flat
    return bets

def get_race_selection():
    st.sidebar.header("Race Selection")
    date = st.sidebar.date_input("Date", datetime.date.today())
    venue = st.sidebar.selectbox("Venue", ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"])
    race_no = st.sidebar.slider("Race No", 1, 12, 1)
    return date, venue, race_no

def format_formations(combos):
    if not combos: return ""
    parsed = [tuple(c.split('-')) for c in combos]
    by_1st = {}
    for c in parsed:
        by_1st.setdefault(c[0], []).append(c)
        
    formations = []
    for f1, items in by_1st.items():
        set_2nd = set(item[1] for item in items)
        set_3rd = set(item[2] for item in items)
        
        cross = []
        for b2 in set_2nd:
            for b3 in set_3rd:
                if f1 != b2 and f1 != b3 and b2 != b3:
                    cross.append((f1, b2, b3))
                    
        if set(cross) == set(items):
            f2_str = "".join(sorted(list(set_2nd)))
            f3_str = "".join(sorted(list(set_3rd)))
            formations.append(f"{f1}-{f2_str}-{f3_str}")
        else:
            by_2nd = {}
            for item in items:
                by_2nd.setdefault(item[1], []).append(item[2])
            
            by_3rd_set = {}
            for f2, f3_list in by_2nd.items():
                f3_tuple = tuple(sorted(list(set(f3_list))))
                by_3rd_set.setdefault(f3_tuple, []).append(f2)
                
            for f3_tuple, f2_list in by_3rd_set.items():
                f2_str = "".join(sorted(f2_list))
                f3_str = "".join(f3_tuple)
                formations.append(f"{f1}-{f2_str}-{f3_str}")
                
    return " / ".join(formations)

def main():
    st.title("🚤 BoatRace AI V3.1+ (Plan B Investment Strategy Engine)")
    st.markdown("本命・穴デュアルモデルを用いた確率＆オッズ連動型自動投資・資金分配エンジン")
    
    # Sidebar
    st.sidebar.header("💰 Investment Config")
    strategy_mode = st.sidebar.radio("Strategy Mode", ["Plan A (全レース参戦)", "Plan B (厳選フィルター)"])
    base_budget = st.sidebar.number_input("Base Budget (Flat Payout)", min_value=0, max_value=10000, value=1000, step=100)
    bonus_budget = st.sidebar.number_input("Bonus Budget (EV Boost)", min_value=0, max_value=10000, value=500, step=100)
    
    debug_mode = st.sidebar.checkbox("Show Debug Info", value=False)
    
    date_dt, venue_name, race_no = get_race_selection()
    date_str = date_dt.strftime("%Y%m%d")
    
    INV_PLACE_CODE = {v: k for k, v in config.PLACE_CODE_TO_NAME.items()}
    place_code = INV_PLACE_CODE.get(venue_name, "01")
    
    models = load_models()
    
    if st.sidebar.button("Analyze & Predict Race", type="primary"):
        if models is None:
            st.error("Models failed to load.")
            return

        with st.spinner("Scraping Live Race Data & Odds..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                scraped_data = loop.run_until_complete(get_realtime_data(date_str, place_code, str(race_no)))
            finally:
                loop.close()
        
        if not scraped_data:
            st.error("Failed to fetch data. Check race schedule or network.")
            return

        try:
            race_info = scraped_data['race_info']
            all_odds = scraped_data['odds_info']
            
            # Create feature input from scraped_data
            entries = race_info['entries']
            before = race_info['before_info']
            
            rows = []
            for idx, row in entries.iterrows():
                bn = int(row['boat_number'])
                rows.append({
                    'race_id': f"{race_info['date']}_{race_info['place']}_{race_info['race_no']}",
                    'boat_number': bn,
                    'racer_id': int(row['racer_id']) if pd.notna(row['racer_id']) else 9999,
                    'motor_rate': float(row['motor_rate']) if pd.notna(row['motor_rate']) else 30.0,
                    'boat_rate': float(row['boat_rate']) if pd.notna(row['boat_rate']) else 30.0,
                    'exhibition_time': float(before['exhibition_times'].get(bn, 6.8)),
                    'exhibition_start_timing': float(before['start_times'].get(bn, 0.20)),
                    'pred_course': int(before['exhibition_entry_courses'].get(bn, bn)),
                    'wind_direction': before.get('wind_direction', 0),
                    'wind_speed': float(before.get('wind_speed', 0.0)) if before.get('wind_speed') is not None else 0.0,
                    'wave_height': float(before.get('wave_height', 0.0)) if before.get('wave_height') is not None else 0.0,
                    'prior_results': row.get('prior_results', ''),
                    'branch': row.get('branch', 'Unknown'),
                    'weight': float(row['weight']) if pd.notna(row['weight']) else 52.0,
                    'nat_win_rate': float(row['nat_win_rate']) if pd.notna(row['nat_win_rate']) else 0.0,
                    'local_win_rate': float(row['loc_win_rate']) if pd.notna(row['loc_win_rate']) else 0.0
                })
            df_race = pd.DataFrame(rows)
            
            df_feat = FeatureEngineer.process(df_race, venue_name, debug_mode=debug_mode)
            
            # Predict
            model_h = models['honmei']
            model_a = models['ana']
            
            feats_h = model_h.feature_name()
            feats_a = model_a.feature_name()
            
            known_cats = ['branch', 'wind_direction', 'venue_code_y', 'class', 'racer_class']
            
            for f in feats_h:
                if f not in df_feat.columns:
                    if f in known_cats:
                        df_feat[f] = '00' 
                        df_feat[f] = df_feat[f].astype('category')
                    else:
                        df_feat[f] = 0.0
                else:
                    if f in known_cats and not pd.api.types.is_categorical_dtype(df_feat[f]):
                        df_feat[f] = df_feat[f].astype(str).astype('category')
                        
            for f in feats_a:
                if f not in df_feat.columns:
                    if f in known_cats:
                        df_feat[f] = '00' 
                        df_feat[f] = df_feat[f].astype('category')
                    else:
                        df_feat[f] = 0.0
                else:
                    if f in known_cats and not pd.api.types.is_categorical_dtype(df_feat[f]):
                        df_feat[f] = df_feat[f].astype(str).astype('category')

            preds_h = model_h.predict(df_feat[feats_h])
            preds_a = model_a.predict(df_feat[feats_a])
            
            df_feat['score_honmei'] = preds_h
            df_feat['score_ana'] = preds_a
            
            scores_h = dict(zip(df_feat['boat_number'], df_feat['score_honmei']))
            scores_a = dict(zip(df_feat['boat_number'], df_feat['score_ana']))
            
            pl_probs, max_p1, prob_gap = calculate_plackett_luce_probs(scores_h)
            
            st.subheader("📊 Race Inference Analytics")
            st.write(f"**Max 1st Prob (P1):** `{max_p1:.4f}` | **Prob Gap:** `{prob_gap:.4f}`")
            
            # 運用モードの切り替え
            if strategy_mode == "Plan A (全レース参戦)":
                st.success("🔥 **Target Race (Plan A)** - 積極参戦モード（全レース購入）で推奨買い目を生成します。")
            else:
                # Plan B Filter Thresholds
                is_target_race = (max_p1 >= 0.49 and prob_gap >= 0.010)
                if is_target_race:
                    st.success("🎯 **Target Race (Plan B)** - Optimal conditions met. Recommended to Bet!")
                else:
                    st.warning("☕ **Skip (Ken)** - Does not meet Plan B confidence thresholds. Recommend watching only.")

            
            selected_combos = select_hybrid_formation_plan_b(pl_probs, scores_a, all_odds)
            bets = calculate_funds_distribution(selected_combos, pl_probs, all_odds, base_budget, bonus_budget)
            
            st.markdown("---")
            if not selected_combos:
                st.info("No combinations selected (Odds too low or no options meet constraints).")
            else:
                st.markdown("### 🎯 Recommended Buying List (Plan B)")
                
                # フォーメーション表示の追加
                formation_str = format_formations(selected_combos)
                st.success(f"**【推奨フォーメーション】** {formation_str}")
                
                df_bets = pd.DataFrame([
                    {
                        'Combo': c,
                        'PL Prob': f"{next(p['prob'] for p in pl_probs if p['combo'] == c):.2%}",
                        'Live Odds': f"{all_odds.get(c, 0.0):.1f}",
                        'Bet Amount (JPY)': bets.get(c, 100),
                        'Est. Return': int(bets.get(c, 100) * all_odds.get(c, 0.0))
                    }
                    for c in selected_combos
                ])
                st.dataframe(df_bets, hide_index=True)
                st.success(f"**Total Investment:** {sum(bets.values())} JPY")
            
            if debug_mode:
                with st.expander("🛠 Raw Data Debug View"):
                    st.write(f"**Engineered Features:**")
                    st.dataframe(df_feat)
                    st.write("**Top Plackett-Luce Probabilities:**")
                    st.write(pl_probs[:10])
                    
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            st.code(traceback.format_exc())
            logger.error(f"Prediction flow failed: {e}")
            return

if __name__ == "__main__":
    main()
