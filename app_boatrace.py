"""
app_boatrace.py
🚤 BOATRACE AI Quantitative Investment System (v3.3)
- ハイブリッドUI: 【自動運用ダッシュボード】 & 【マニュアル推論】
- Supabase (PostgreSQL 17) リアルタイム連携 (db_manager)
- Gatekeeper (85th percentile 相対評価 / Platt Scaling)
- Extractor (Odds Residual + OddsNormalizer)
- 会場クラスタ別 Benter展開 (d2, d3 自動最適化)
- Markowitz / Fractional Kelly ポートフォリオ最適化
"""

import os
import sys

# Force single thread to prevent Streamlit Cloud crashes (OpenMP)
os.environ['OMP_NUM_THREADS'] = '1'

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import requests
from bs4 import BeautifulSoup
import datetime
import re
import time
import itertools
import json

from odds_normalizer import probs_to_init_scores
from probability_calibration import (
    calculate_benter_probs,
    calculate_plackett_luce_probs,
    load_probability_config,
    get_default_calibrator,
    get_cluster_benter_params,
    load_benter_cluster_config
)
from portfolio_optimizer import PortfolioOptimizer, load_correlation_mask
import db_manager

# Page Config
st.set_page_config(
    page_title="BOATRACE AI Dual Quant System",
    page_icon="🚤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Dark Theme with sleek accents)
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .investment-card {
        background: linear-gradient(135deg, rgba(0, 230, 118, 0.08), rgba(0, 200, 83, 0.02));
        border: 1.5px solid #00E676;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 6px 20px rgba(0, 230, 118, 0.15);
    }
    .badge-win {
        background-color: #00c853;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-skip {
        background-color: #ff9100;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .badge-cluster1 {
        background-color: #d50000;
        color: white;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'
MODEL_ANA_PATH = 'model_ana.txt'
DATA_DIR = 'app_data'
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}
CLUSTER_1_VENUES = [2, 3, 4, 14, 22]  # 戸田02, 江戸川03, 平和島04, 鳴門14, 福岡22

VENUE_MAP = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
}


# =====================================================================
# 1. スクレイパー & 特徴量エンジニアリング
# =====================================================================

class BoatRaceScraper:
    @staticmethod
    def get_soup(url):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"データ取得エラー: {e}")
                    return None
                time.sleep(1)
        return None

    @staticmethod
    def parse_float(text):
        try:
            return float(re.search(r'([\d\.]+)', text).group(1))
        except Exception:
            return 0.0

    @staticmethod
    def get_odds(date_str, venue_code, race_no):
        jcd = f"{int(venue_code):02d}"
        url = f"https://www.boatrace.jp/owpc/pc/race/odds3t?rno={race_no}&jcd={jcd}&hd={date_str}"
        soup = BoatRaceScraper.get_soup(url)
        odds_data = {}
        if soup:
            try:
                target_tbody = soup.select_one("tbody.is-p3-0")
                if not target_tbody:
                    target_tbody = soup.select_one("div.table1 table tbody")
                if not target_tbody:
                    return {}

                first_places = []
                header_row = target_tbody.parent.select_one("thead tr")
                if header_row:
                    th_tags = header_row.find_all('th', recursive=False)
                    for th in th_tags:
                        if th.has_attr('class') and any('is-boatColor' in c for c in th['class']):
                            th_text = th.get_text(strip=True)
                            num_match = re.match(r'^\s*(\d+)', th_text)
                            if num_match:
                                num = num_match.group(1)
                                if num not in first_places:
                                    first_places.append(num)
                                    if len(first_places) == 6:
                                        break
                
                if len(first_places) != 6:
                    first_places = [str(i) for i in range(1, 7)]

                rows = target_tbody.find_all('tr', recursive=False)
                num_cols = len(first_places)
                current_boat2 = [''] * num_cols
                rowspan_remaining = [0] * num_cols

                for r_idx, row in enumerate(rows):
                    cells = row.find_all('td', recursive=False)
                    cell_ptr = 0
                    for col_idx in range(num_cols):
                        first_boat = first_places[col_idx]
                        boat2 = ""
                        boat3 = ""
                        odds_val = None
                        
                        try:
                            if rowspan_remaining[col_idx] > 0:
                                rowspan_remaining[col_idx] -= 1
                                if cell_ptr + 1 < len(cells):
                                    boat2 = current_boat2[col_idx]
                                    boat3 = cells[cell_ptr].get_text(strip=True)
                                    odds_txt = cells[cell_ptr + 1].get_text(strip=True).replace('倍', '').replace(',', '')
                                    try: odds_val = float(odds_txt)
                                    except Exception: pass
                                    cell_ptr += 2
                                else: continue
                            else:
                                if cell_ptr >= len(cells): break
                                current_cell = cells[cell_ptr]
                                if current_cell.has_attr('rowspan'):
                                    boat2_text = current_cell.get_text(strip=True)
                                    if boat2_text.isdigit():
                                        current_boat2[col_idx] = boat2_text
                                        boat2 = boat2_text
                                        try: rowspan_remaining[col_idx] = max(0, int(current_cell['rowspan']) - 1)
                                        except Exception: rowspan_remaining[col_idx] = 0
                                        if cell_ptr + 2 < len(cells):
                                            boat3 = cells[cell_ptr+1].get_text(strip=True)
                                            odds_txt = cells[cell_ptr+2].get_text(strip=True).replace('倍', '').replace(',', '')
                                            try: odds_val = float(odds_txt)
                                            except Exception: pass
                                            cell_ptr += 3
                                        else:
                                            cell_ptr += 1
                                            continue
                                    else:
                                        cell_ptr += 1
                                        continue
                                else:
                                    cell_ptr += 1
                                    continue
                            
                            if boat2.isdigit() and boat3.isdigit() and first_boat != boat2 and first_boat != boat3 and boat2 != boat3 and odds_val is not None and odds_val > 0:
                                odds_data[f"{first_boat}-{boat2}-{boat3}"] = odds_val
                        except Exception: pass
            except Exception: pass
        return odds_data

    @staticmethod
    def get_race_data(date_str, venue_code, race_no):
        jcd = f"{int(venue_code):02d}"
        url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date_str}"
        url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_no}&jcd={jcd}&hd={date_str}"
        
        soup_before = BoatRaceScraper.get_soup(url_before)
        soup_list = BoatRaceScraper.get_soup(url_list)
        
        if not soup_before or not soup_list:
            return None
            
        weather = {'wind_direction': 0, 'wind_speed': 0.0, 'wave_height': 0.0}
        try:
            w = soup_before.select_one("div.weather1_body")
            if w:
                ws = w.select_one(".is-wind span.weather1_bodyUnitLabelData")
                if ws: weather['wind_speed'] = BoatRaceScraper.parse_float(ws.text)
                wh = w.select_one(".is-wave span.weather1_bodyUnitLabelData")
                if wh: weather['wave_height'] = BoatRaceScraper.parse_float(wh.text)
                wd = w.select_one(".is-windDirection p")
                if wd:
                    cls = wd.get('class', [])
                    d = next((c for c in cls if c.startswith('is-wind') and c != 'is-windDirection'), None)
                    if d: weather['wind_direction'] = int(re.sub(r'\D', '', d))
        except Exception: pass

        boat_before = {}
        try:
            for i, tb in enumerate(soup_before.select("table.is-w748 tbody")):
                tds = tb.select("td")
                ex_val = None
                if len(tds) >= 5:
                    txt = tds[4].get_text(strip=True)
                    if txt and txt != '\xa0':
                        try: ex_val = float(re.search(r'([\d\.]+)', txt).group(1))
                        except Exception: pass
                if ex_val is not None:
                    if (i+1) not in boat_before: boat_before[i+1] = {}
                    boat_before[i+1]['ex_time'] = ex_val
            
            for idx, row in enumerate(soup_before.select("table.is-w238 tbody tr")):
                bn_span = row.select_one("span.table1_boatImage1Number")
                if bn_span:
                    b = int(bn_span.text.strip())
                    pred_c = idx + 1
                    st_span = row.select_one("span.table1_boatImage1Time")
                    val = 0.20
                    if st_span:
                        txt_raw = st_span.text.strip()
                        if 'L' in txt_raw: val = 1.0
                        elif 'F' in txt_raw:
                            try:
                                sub = txt_raw.replace('F', '')
                                val = -float(sub)
                            except Exception: val = -0.05
                        else:
                            val = BoatRaceScraper.parse_float(txt_raw)
                            
                    if b not in boat_before: boat_before[b] = {}
                    boat_before[b]['st'] = val
                    boat_before[b]['pred_course'] = pred_c
        except Exception: pass

        rows = []
        try:
            for i, tb in enumerate(soup_list.select("tbody.is-fs12")):
                bn = i + 1
                if bn > 6: break
                if bn not in boat_before or 'ex_time' not in boat_before[bn]:
                    continue
                
                racer_id = 9999
                try: 
                    txt = tb.select("td")[2].select_one("div").get_text()
                    racer_id = int(re.search(r'(\d{4})', txt).group(1))
                except Exception: pass

                branch = 'Unknown'
                weight = 52.0
                try:
                    td2 = tb.select("td")[2]
                    txt_full = td2.get_text(" ", strip=True)
                    match_w = re.search(r'(\d{2}\.\d)kg', txt_full)
                    if match_w: weight = float(match_w.group(1))
                    prefectures = r"(群馬|埼玉|東京|福井|静岡|愛知|三重|滋賀|大阪|兵庫|徳島|香川|岡山|広島|山口|福岡|佐賀|長崎)"
                    m = re.search(prefectures, txt_full)
                    if m: branch = m.group(1)
                except Exception: pass

                nat_win_rate = 0.0
                local_win_rate = 0.0
                try:
                    col3_txt = tb.select("td")[3].get_text(" ", strip=True)
                    clean_txt = re.sub(r'[FLK]\d+', '', col3_txt) 
                    nums = re.findall(r'(\d+(?:\.\d+)?)', clean_txt)
                    if len(nums) >= 5:
                        nat_win_rate = float(nums[1])
                        local_win_rate = float(nums[3])
                    elif len(nums) >= 4:
                        nat_win_rate = float(nums[0])
                        local_win_rate = float(nums[2])
                except Exception: pass

                prior_results = ""
                try:
                    rank_row = tb.select_one("tr.is-fBold")
                    if rank_row:
                        res_texts = [td.get_text(strip=True) for td in rank_row.select("td")]
                        cleaned_res = []
                        for t in res_texts:
                            if not t: continue
                            t_norm = t.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
                            if re.match(r'^[1-6FLKS欠失転不]$', t_norm):
                                cleaned_res.append(t_norm)
                        prior_results = " ".join(cleaned_res)
                except Exception: pass

                tds = tb.select("td")
                motor = 30.0
                try:
                    txt = tds[6].get_text(" ", strip=True).replace('%', '')
                    parts = txt.split()
                    if len(parts) >= 2: motor = float(parts[1])
                    else: motor = float(parts[0])
                except Exception: pass
                
                boat = 30.0
                try:
                    if len(tds) > 7:
                        txt = tds[7].get_text(" ", strip=True).replace('%', '')
                        parts = txt.split()
                        if len(parts) >= 2: boat = float(parts[1])
                        else: boat = float(parts[0])
                except Exception: pass
                
                row = {
                    'race_id': f"{date_str}_{venue_code}_{race_no}",
                    'boat_number': bn,
                    'racer_id': racer_id,
                    'motor_rate': motor,
                    'boat_rate': boat,
                    'exhibition_time': boat_before[bn]['ex_time'],
                    'exhibition_start_timing': boat_before.get(bn, {}).get('st', 0.20),
                    'pred_course': boat_before.get(bn, {}).get('pred_course', bn),
                    'wind_direction': weather['wind_direction'],
                    'wind_speed': weather['wind_speed'],
                    'wave_height': weather['wave_height'],
                    'prior_results': prior_results,
                    'branch': branch,
                    'weight': weight,
                    'nat_win_rate': nat_win_rate,
                    'local_win_rate': local_win_rate
                }
                rows.append(row)
        except Exception as e:
            st.error(f"出走表パースエラー: {e}")
            return None
            
        return pd.DataFrame(rows)


def add_advanced_features(df):
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
    prev_sts = df['corrected_st'].shift(1)
    df['inner_st_gap_corrected'] = df['corrected_st'] - prev_sts
    df.loc[df['boat_number'] == 1, 'inner_st_gap_corrected'] = 0.0
    
    if 'motor_rate' in df.columns and 'exhibition_time' in df.columns:
        df['motor_rank'] = df.groupby('race_id')['motor_rate'].rank(ascending=False, method='min')
        df['tenji_rank'] = df.groupby('race_id')['exhibition_time'].rank(ascending=True, method='min')
        df['motor_gap'] = df['motor_rank'] - df['tenji_rank']
    else:
        df['motor_gap'] = 0.0
        
    if 'venue_course_1st_rate' in df.columns and 'nat_win_rate' in df.columns:
        df['specialist_score'] = df['venue_course_1st_rate'] - df['nat_win_rate']
    else:
        df['specialist_score'] = 0.0
        
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
        
    bias_path = 'app_data/venue_frame_bias.csv'
    if os.path.exists(bias_path):
        try:
            bias_df = pd.read_csv(bias_path)
            bias_df['venue_code'] = bias_df['venue_code'].astype(str).str.zfill(2)
            bias_df['boat_number'] = bias_df['boat_number'].astype(int)
            
            venue_map_codes = {
                '桐生': '01', '戸田': '02', '江戸川': '03', '平和島': '04', '多摩川': '05',
                '浜名湖': '06', '蒲郡': '07', '常滑': '08', '津': '09', '三国': '10',
                'びわこ': '11', '住之江': '12', '尼崎': '13', '鳴門': '14', '丸亀': '15',
                '児島': '16', '宮島': '17', '徳山': '18', '下関': '19', '若松': '20',
                '芦屋': '21', '福岡': '22', '唐津': '23', '大村': '24'
            }
            if 'venue_name' in df.columns:
                df['temp_venue_code'] = df['venue_name'].map(venue_map_codes).fillna('00')
                df = df.merge(bias_df, left_on=['temp_venue_code', 'boat_number'], right_on=['venue_code', 'boat_number'], how='left')
                df.drop(columns=['temp_venue_code', 'venue_code'], inplace=True, errors='ignore')
                df['venue_frame_win_rate'] = df.get('venue_frame_win_rate', pd.Series([0.16]*len(df))).fillna(0.16)
        except Exception:
            df['venue_frame_win_rate'] = 0.16
    else:
        df['venue_frame_win_rate'] = 0.16
        
    return df


class FeatureEngineer:
    @staticmethod
    def process(df, venue_name):
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

            venue_map_rev = {v: k for k, v in VENUE_MAP.items()}
            df['venue_code_int'] = df['venue_name'].map(venue_map_rev).fillna(0).astype(int)
            df['venue_code_y'] = df['venue_code_int'].astype(str).str.zfill(2)
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
        
        required_cols = ['makuri_count', 'nige_count', 'sashi_count', 'nat_win_rate', 'course_run_count', 'local_win_rate']
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
            except Exception: pass
            return 3.5
            
        df['series_avg_rank'] = df['prior_results'].apply(parse_prior)
        df['makuri_rate'] = df['makuri_count'] / df['course_run_count'].replace(0, 1)
        df['nige_rate'] = df['nige_count'] / df['course_run_count'].replace(0, 1)

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
        if std_t == 0 or np.isnan(std_t): std_t = 1.0
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
        
        wind_map = {
            1: '北', 2: '北北東', 3: '北東', 4: '東北東', 5: '東', 6: '東南東', 7: '南東', 8: '南南東',
            9: '南', 10: '南南西', 11: '南西', 12: '西南西', 13: '西', 14: '西北西', 15: '北西', 16: '北北西'
        }
        if pd.api.types.is_numeric_dtype(df['wind_direction']):
            df['wind_direction'] = df['wind_direction'].map(wind_map).fillna(df['wind_direction']).astype(str).replace('nan', '')

        for col in df.columns:
            if col not in ['race_id', 'race_date', 'venue_name', 'prior_results', 'wind_direction', 'branch', 'class', 'racer_class', 'venue_code_y']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


def prepare_features_for_model(df_feat, model):
    feats = model.feature_name()
    pandas_cats = model.pandas_categorical
    df_out = pd.DataFrame(index=df_feat.index)
    
    cat_cols_map = {}
    if pandas_cats:
        cat_candidates = ['branch', 'venue_code_y', 'wind_direction', 'class', 'racer_class']
        actual_cat_cols = [f for f in feats if f in cat_candidates]
        for i, col_name in enumerate(actual_cat_cols):
            if i < len(pandas_cats):
                cat_cols_map[col_name] = pandas_cats[i]
                
    for f in feats:
        if f in cat_cols_map:
            cat_list = cat_cols_map[f]
            if f in df_feat.columns:
                val_series = df_feat[f].astype(str)
            elif f == 'venue_code_y' and 'venue_code_int' in df_feat.columns:
                val_series = df_feat['venue_code_int'].astype(str).str.zfill(2)
            elif f == 'venue_code_y' and 'temp_venue_code' in df_feat.columns:
                val_series = df_feat['temp_venue_code'].astype(str).str.zfill(2)
            else:
                val_series = pd.Series([cat_list[0]] * len(df_feat), index=df_feat.index)
            df_out[f] = pd.Categorical(val_series, categories=cat_list)
        else:
            if f in df_feat.columns:
                df_out[f] = pd.to_numeric(df_feat[f], errors='coerce').fillna(0.0).astype(float)
            elif f == 'syn_win_rate':
                df_out[f] = 0.0
            else:
                df_out[f] = 0.0
                
    return df_out


# =====================================================================
# 2. Sidebar Navigation & Global Controls
# =====================================================================

st.sidebar.title("🚤 BOATRACE AI Dual Quant")

# 操作モード切替
app_mode = st.sidebar.radio(
    "🧭 操作モード選択",
    ["📊 自動運用ダッシュボード (Auto Dashboard)", "🎯 マニュアル推論 (Manual Mode)"],
    index=0
)

st.sidebar.markdown("---")

# DB 接続バッジ
with db_manager.get_db_connection() as db:
    if db.is_postgres:
        st.sidebar.success("🟢 Supabase (PostgreSQL 17) 接続中")
    else:
        st.sidebar.info("📁 SQLite (Local DB) 接続中")

if st.sidebar.button("🧹 キャッシュクリア (Clear Cache)", use_container_width=True):
    st.cache_data.clear()
    st.sidebar.success("キャッシュをクリアしました！")


# =====================================================================
# 3. Mode A: 📊 自動運用ダッシュボード (Auto Dashboard)
# =====================================================================

if app_mode == "📊 自動運用ダッシュボード (Auto Dashboard)":
    st.title("📊 自動運用モニタリングダッシュボード")
    st.caption("`auto_trader.py` による自動推論・Gatekeeper判定・投資GOサインのリアルタイム集計")

    # 上部コントロールバー
    col_c1, col_c2, col_c3 = st.columns([2, 1, 1])
    with col_c1:
        today_date_str = datetime.date.today().strftime('%Y%m%d')
        date_filter_mode = st.selectbox(
            "📅 抽出期間 (Date Filter)",
            [f"本日 ({today_date_str})", "全期間 (All History)"],
            index=0
        )
        selected_date = today_date_str if "本日" in date_filter_mode else None
        
    with col_c2:
        if st.button("🔄 最新データ再取得", use_container_width=True):
            st.rerun()
            
    with col_c3:
        if st.button("🧪 モック推論生成", help="テスト用の模擬投資GOレースを生成してDBへ保存します", use_container_width=True):
            import auto_trader
            with st.spinner("模擬レース推論を実行中..."):
                auto_trader.evaluate_mock_race(venue_code=18, race_no=10, dry_run=True)
            st.success("テスト用モックレースを生成しました！")
            st.rerun()

    # KPI メトリクス
    stats = db_manager.get_dashboard_stats(selected_date)
    
    st.markdown("### 📈 運用パフォーマンス サマリー")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85em; color: #888;">🏟️ 分析完了レース数</div>
            <div style="font-size: 1.8em; font-weight: 700;">{stats['total_evaluated']:,} <span style="font-size:0.5em; font-weight:400;">R</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85em; color: #888;">🛡️ Gatekeeper 通過率 (P1 >= 74.38%)</div>
            <div style="font-size: 1.8em; font-weight: 700; color: #00E5FF;">{stats['gatekeeper_passed']:,} <span style="font-size:0.5em; font-weight:400;">R ({stats['gatekeeper_rate']:.1%})</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85em; color: #888;">🚀 投資GOサイン 点灯数</div>
            <div style="font-size: 1.8em; font-weight: 700; color: #00E676;">{stats['investment_go']:,} <span style="font-size:0.5em; font-weight:400;">R</span></div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size: 0.85em; color: #888;">💰 推奨投資総額 (SLSQP最適配分)</div>
            <div style="font-size: 1.8em; font-weight: 700; color: #FFD600;">{stats['total_recommended_bet']:,} <span style="font-size:0.5em; font-weight:400;">円</span></div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # フィルター設定
    f1, f2 = st.columns([1, 1])
    with f1:
        status_filter = st.selectbox(
            "🔍 ステータス絞り込み",
            ["all", "investment_go", "gatekeeper_passed", "gatekeeper_skipped", "skipped_cluster1", "no_value_bets"],
            format_func=lambda x: {
                "all": "全ステータス (All)",
                "investment_go": "🚀 投資GOサインのみ (Investment GO)",
                "gatekeeper_passed": "🛡️ Gatekeeper通過のみ (Passed)",
                "gatekeeper_skipped": "☕ Gatekeeper未達見送り (Skipped)",
                "skipped_cluster1": "🛑 難水面システム除外 (Cluster 1)",
                "no_value_bets": "🔍 EV条件未達見送り (No Value)"
            }.get(x, x)
        )
    with f2:
        venue_filter = st.selectbox(
            "🏟️ 開催場絞り込み",
            ["all"] + list(VENUE_MAP.values()),
            format_func=lambda x: "全会場 (All Venues)" if x == "all" else x
        )

    # 投資GOサイン ピックアップセクション
    go_races = db_manager.get_all_predictions_with_bets(
        date_str=selected_date,
        status_filter="investment_go",
        venue_filter=venue_filter,
        limit=20
    )

    st.markdown("### 🚀 【投資GOサイン】本日の勝負レース")
    if go_races:
        for r in go_races:
            rid = r['race_id']
            vname = r['venue_name']
            rno = r['race_no']
            dtime = r.get('deadline_time', '--:--')
            p1 = r.get('max_p1', 0.0)
            top_b = r.get('top_boat', 1)
            gap = r.get('prob_gap', 0.0)
            cname = r.get('cluster_name', '標準水面')
            cid = r.get('cluster_id', 2)
            bets = r.get('bets', [])
            tbet = r.get('total_bet', 0)
            max_ret = r.get('max_return', 0)
            
            st.markdown(f"""
            <div class="investment-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="font-size: 1.25em; font-weight: 700;">
                        🚀 {vname} {rno}R <span style="font-size:0.8em; color:#aaa; font-weight:normal;">(締切: {dtime})</span>
                    </div>
                    <div>
                        <span class="badge-win">投資GOサイン点灯</span>
                    </div>
                </div>
                <div style="font-size: 0.9em; color: #ccc; margin-bottom: 12px;">
                    🛡️ <b>Gatekeeper P1:</b> {p1:.1%} ({top_b}号艇本命 / 2位差: {gap:+.1%}) &nbsp;|&nbsp; 
                    🏟️ <b>水面クラスタ:</b> {cname} (Cluster {cid}) &nbsp;|&nbsp; 
                    💰 <b>投資総額:</b> {tbet:,} 円 (最高払戻見込: {max_ret:,} 円)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if bets:
                df_bets = pd.DataFrame(bets)[['combination', 'bet_amount', 'prob', 'odds', 'ev', 'expected_return']]
                df_bets.columns = ['買い目', '推奨金額 (円)', 'Benter確率', '実オッズ', 'EV (期待値)', '払戻見込 (円)']
                df_bets['推奨金額 (円)'] = df_bets['推奨金額 (円)'].apply(lambda x: f"{x:,} 円")
                df_bets['Benter確率'] = df_bets['Benter確率'].apply(lambda x: f"{x:.2%}")
                df_bets['実オッズ'] = df_bets['実オッズ'].apply(lambda x: f"{x:.1f} 倍")
                df_bets['EV (期待値)'] = df_bets['EV (期待値)'].apply(lambda x: f"{x:.2f}")
                df_bets['払戻見込 (円)'] = df_bets['払戻見込 (円)'].apply(lambda x: f"{x:,} 円")
                st.dataframe(df_bets, use_container_width=True, hide_index=True)
            st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.info("💡 現在、投資条件（水面適格 × Gatekeeper通過 × EV >= 1.25）を満たした勝負レースはありません。自動ワーカー (`auto_trader.py`) が発走5分前にリアルタイム監視待機中です。")

    st.markdown("---")

    # 全推論履歴テーブル
    st.markdown("### 📋 全レース推論 & 判定履歴一覧")
    all_races = db_manager.get_all_predictions_with_bets(
        date_str=selected_date,
        status_filter=status_filter,
        venue_filter=venue_filter,
        limit=100
    )

    if all_races:
        history_rows = []
        for r in all_races:
            p1_val = f"{r['max_p1']:.1%}" if r.get('max_p1') is not None else "-"
            top_b_val = f"{r['top_boat']}号艇" if r.get('top_boat') is not None else "-"
            gap_val = f"{r['prob_gap']:+.1%}" if r.get('prob_gap') is not None else "-"
            
            status_badge = {
                'investment_go': '🚀 投資GO',
                'mock_investment_go': '🧪 投資GO(Mock)',
                'gatekeeper_skipped': '☕ GK未達',
                'skipped_cluster1': '🛑 難水面除外',
                'no_value_bets': '🔍 EV未達'
            }.get(r['status'], r['status'])
            
            history_rows.append({
                'レースID': r['race_id'],
                '日付': r['race_date'],
                '会場': r['venue_name'],
                'R': f"{r['race_no']}R",
                '締切': r.get('deadline_time', '-'),
                '本命艇': top_b_val,
                'Gatekeeper P1': p1_val,
                '2位差': gap_val,
                '通過': "✅ 通過" if r['gatekeeper_passed'] else "❌ 未達",
                'クラスタ': r.get('cluster_name', '-'),
                '判定ステータス': status_badge,
                '買い目数': len(r.get('bets', [])),
                '投資総額': f"{r.get('total_bet', 0):,} 円" if r.get('total_bet', 0) > 0 else "-",
                '評価日時': str(r['created_at'])[:19]
            })
            
        df_history = pd.DataFrame(history_rows)
        st.dataframe(df_history, use_container_width=True, hide_index=True)
    else:
        st.write("該当する推論履歴データは見つかりませんでした。")

    # Discord 通知ログ エキスパンダー
    with st.expander("📨 Discord Webhook 配信ログ履歴 (最新20件)"):
        logs = db_manager.get_notification_logs(limit=20)
        if logs:
            for log_item in logs:
                st.markdown(f"**[{str(log_item['created_at'])[:19]}] {log_item['title']}** (Status: `{log_item['status']}`)")
                if log_item.get('message_payload'):
                    st.json(log_item['message_payload'])
                st.markdown("---")
        else:
            st.write("配信ログはまだありません。")


# =====================================================================
# 4. Mode B: 🎯 マニュアル推論 (Manual Mode)
# =====================================================================

else:
    st.title("🎯 マニュアル推論 & ポートフォリオ最適化")
    st.caption("任意のレースを指定し、Gatekeeper閾値やBenterパラメーターを手動調整して即時分析")

    today = datetime.date.today()
    
    # サイドバー操作
    st.sidebar.header("🎯 対象レース指定")
    target_date = st.sidebar.date_input("開催日 (Date)", today)
    venue_code = st.sidebar.selectbox("開催場 (Venue)", list(VENUE_MAP.keys()), format_func=lambda x: f"{x:02d}: {VENUE_MAP[x]}")
    venue_name = VENUE_MAP[venue_code]
    race_no = st.sidebar.selectbox("レース番号 (Race No)", range(1, 13))

    debug_mode = st.sidebar.checkbox("デバッグ情報を表示", value=False)

    st.sidebar.markdown("---")
    st.sidebar.header("🛡️ Gatekeeper スクリーニング")
    gatekeeper_th = st.sidebar.slider(
        "信頼度判定カットオフ (P1 閾値)",
        min_value=0.40,
        max_value=0.90,
        value=0.7438,
        step=0.005,
        format="%.4f",
        help="Gatekeeper（model_honmei.txt）の1着確率P1がこの値以上のレースのみを勝負レースとして抽出します（黄金ベースライン: 74.38%）。"
    )

    st.sidebar.markdown("---")
    st.sidebar.header("🏟️ 水面クラスタ & Benter設定")
    cluster_d2, cluster_d3, cluster_id, cluster_name = get_cluster_benter_params(venue_code)

    if venue_code in CLUSTER_1_VENUES:
        st.sidebar.warning(
            f"⚠️ **水面区分:** {cluster_name} (Cluster {cluster_id})\n\n"
            f"🚨 **難水面除外システム作動中**\n\n"
            f"この会場（戸田・江戸川・平和島・鳴門・福岡）は波乱リスクが高いため、投資純度保護の観点から推論・投資がスキップされます。"
        )
    else:
        st.sidebar.info(
            f"**水面区分:** {cluster_name} (Cluster {cluster_id})\n\n"
            f"**最適パラメーター:** `d2 = {cluster_d2:.2f}`, `d3 = {cluster_d3:.2f}`"
        )

    use_cluster_auto = st.sidebar.checkbox("会場クラスタ別 最適Benter値を自動適用", value=True)
    if use_cluster_auto:
        d2_effective = cluster_d2
        d3_effective = cluster_d3
        st.sidebar.caption(f"⚡ 自動適用中: d2={d2_effective:.2f}, d3={d3_effective:.2f}")
    else:
        d2_effective = st.sidebar.slider("2着 減衰パラメーター (d2)", min_value=0.05, max_value=1.50, value=float(cluster_d2), step=0.05)
        d3_effective = st.sidebar.slider("3着 減衰パラメーター (d3)", min_value=0.05, max_value=1.50, value=float(cluster_d3), step=0.05)

    st.sidebar.markdown("---")
    st.sidebar.header("💰 資金配分戦略 (Portfolio Strategy)")

    strategy_choice = st.sidebar.radio(
        "資金配分モデル",
        ["固定ウェイト上限 (Max 5%)", "クォーター・ケリー (Fractional Kelly f=0.25)"],
        index=0,
        help="固定ウェイト上限: レース総投資額を一律上限5%（買い目上限2%）に制限。\nクォーター・ケリー: 候補買い目のエッジ合計に応じレース投資比率を動的決定（上限10%）。"
    )

    bankroll = st.sidebar.number_input("想定バンクロール (円)", min_value=10000, max_value=10000000, value=100000, step=10000)
    risk_aversion = st.sidebar.slider("リスク回避度 (λ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
    min_ev = st.sidebar.number_input("最小期待値閾値 (Min EV)", min_value=1.00, max_value=3.00, value=1.25, step=0.05)
    max_odds = st.sidebar.number_input("最大オッズ上限 (Max Odds)", min_value=5.0, max_value=200.0, value=30.0, step=5.0)

    # 推論実行ボタン
    if st.button("🚀 レース分析 & 最適買い目算出 (Analyze Race)", type="primary", use_container_width=True):
        st.session_state['run_analysis'] = True

    if st.session_state.get('run_analysis', False):
        date_str = target_date.strftime("%Y%m%d")
        race_id = f"{date_str}_{venue_code}_{race_no}"
        
        # 難水面（Cluster 1）除外チェック
        if venue_code in CLUSTER_1_VENUES:
            st.error(f"🛑 【参戦見送り】{venue_name} は難水面（Cluster 1: 波乱場）に該当するため、システム保護により推論・投資を即時スキップしました。")
            st.info("💡 黄金ベースラインのバックテスト検証に基づき、イン勝率が安定した標準水面（Cluster 2）およびイン超強水面（Cluster 0）に投資資金を集中させます。")
            
            try:
                db_manager.save_race_prediction(
                    race_id=race_id, race_date=date_str, venue_code=venue_code,
                    venue_name=venue_name, race_no=race_no, deadline_time="マニュアル",
                    top_boat=None, max_p1=None, prob_gap=None, gatekeeper_passed=False,
                    cluster_id=cluster_id, cluster_name=cluster_name, status="skipped_cluster1"
                )
            except Exception: pass
            st.stop()
            
        with st.spinner(f"{venue_name} {race_no}R の直前オッズ・出走表・展示タイムを取得中..."):
            df_race = BoatRaceScraper.get_race_data(date_str, venue_code, race_no)
            all_odds = BoatRaceScraper.get_odds(date_str, venue_code, race_no)
            
        if df_race is None or df_race.empty:
            st.warning("出走表または展示データの取得に失敗しました。展示開始前、発売中止、または非開催日の可能性があります。")
            st.stop()
            
        if not all_odds:
            st.warning("オッズデータの取得に失敗しました。オッズ発表前か、レースが終了している可能性があります。")
            st.stop()
            
        # オッズを Supabase に保存
        try:
            db_manager.save_odds_batch(race_id, all_odds)
        except Exception: pass

        # 特徴量生成 & 欠場艇判定
        df_feat = FeatureEngineer.process(df_race, venue_name)
        active_boats = df_feat['boat_number'].tolist()
        absent_boats = sorted(list(set(range(1, 7)) - set(active_boats)))
        
        if absent_boats:
            absent_str = "、".join([f"{b}号艇" for b in absent_boats])
            st.warning(f"📢 **【欠場情報】** {absent_str} は欠場（出走除外）のため、{len(active_boats)}艇立てとして動的推論を実施します。")
            
        # オッズ合成勝率
        syn_dict = {b: 0.0 for b in active_boats}
        for combo, o_val in all_odds.items():
            try:
                parts = combo.split('-')
                b1 = int(parts[0])
                if o_val > 0 and b1 in syn_dict:
                    syn_dict[b1] += (1.0 / o_val)
            except Exception: pass
            
        total_syn = sum(syn_dict.values())
        if total_syn > 0:
            p_norm_syn_dict = {b: syn_dict[b] / total_syn for b in active_boats}
        else:
            p_norm_syn_dict = {b: 1.0 / len(active_boats) for b in active_boats}
            
        df_feat['syn_win_rate'] = df_feat['boat_number'].map(p_norm_syn_dict).fillna(1.0 / len(active_boats))
        df_feat['init_score'] = probs_to_init_scores(df_feat['syn_win_rate'].to_numpy())

        # Gatekeeper 推論
        model_h = lgb.Booster(model_file=MODEL_HONMEI_PATH)
        df_h = prepare_features_for_model(df_feat, model_h)
        df_feat['score_honmei'] = model_h.predict(df_h)
        calibrator = get_default_calibrator('platt')
        scores_h_dict = dict(zip(df_feat['boat_number'], df_feat['score_honmei']))
        p1_dict_honmei = calibrator.calibrate_scores(scores_h_dict)
        
        sorted_p1 = sorted(p1_dict_honmei.items(), key=lambda x: x[1], reverse=True)
        top_boat, max_p1 = sorted_p1[0]
        prob_gap = (max_p1 - sorted_p1[1][1]) if len(sorted_p1) > 1 else 0.0
        
        # 1. Gatekeeper 判定表示
        st.subheader("🛡️ 第1段階: Gatekeeper スクリーニング判定")
        col_g1, col_g2, col_g3 = st.columns([1, 1, 1])
        with col_g1:
            st.metric("本命艇 (Top Boat)", f"{top_boat} 号艇")
        with col_g2:
            st.metric("Gatekeeper 1着確率 (P1)", f"{max_p1:.2%}", delta=f"閾値差: {max_p1 - gatekeeper_th:+.2%}")
        with col_g3:
            st.metric("2位との勝率差 (Prob Gap)", f"{prob_gap:+.2%}")
            
        if max_p1 < gatekeeper_th:
            st.warning(f"☕ **【参戦見送り】** Gatekeeperの信頼度スコア ({max_p1:.2%}) が基準値 ({gatekeeper_th:.2%}) に満たないため、このレースは投資対象外（No Bet）です。")
            try:
                db_manager.save_race_prediction(
                    race_id=race_id, race_date=date_str, venue_code=venue_code,
                    venue_name=venue_name, race_no=race_no, deadline_time="マニュアル",
                    top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=False,
                    cluster_id=cluster_id, cluster_name=cluster_name, status="gatekeeper_skipped"
                )
            except Exception: pass
            st.stop()
            
        st.success(f"🎯 **【Gatekeeper 通過】** 信頼度スコア {max_p1:.2%} ≧ {gatekeeper_th:.2%}！ 勝負レース判定。第2段階（Extractor & ポートフォリオ最適化）を実行します。")
        
        # Extractor 推論
        model_r = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
        df_r = prepare_features_for_model(df_feat, model_r)
        raw_res = model_r.predict(df_r, raw_score=True)
        total_logits = raw_res + df_feat['init_score'].to_numpy()
        p_raw_res = 1.0 / (1.0 + np.exp(-np.clip(total_logits, -30, 30)))
        p_norm_res = p_raw_res / np.sum(p_raw_res)
        p1_dict_residual = dict(zip(df_feat['boat_number'], p_norm_res))

        # Benter 展開
        benter_probs, _, _ = calculate_benter_probs(
            p1_dict_residual,
            d2=d2_effective,
            d3=d3_effective,
            calibration_method='direct'
        )
        benter_probs_dict = {p['combo']: p['prob'] for p in benter_probs}

        # ポートフォリオ最適化
        optimizer = PortfolioOptimizer()
        kelly_param = 0.25 if "ケリー" in strategy_choice else None
        
        bets = optimizer.optimize_funds(
            probabilities=benter_probs_dict,
            odds=all_odds,
            bankroll=float(bankroll),
            risk_aversion=float(risk_aversion),
            max_exposure=0.05,
            max_concentration=0.02,
            min_ev=float(min_ev),
            max_odds=float(max_odds),
            kelly_fraction=kelly_param
        )
        
        # 結果表示
        st.markdown("---")
        st.subheader("🚀 第2段階: ポートフォリオ最適化 結果")
        
        if bets:
            total_bet = sum(bets.values())
            max_ret = max([amt * all_odds.get(c, 0.0) for c, amt in bets.items()])
            
            r1, r2, r3 = st.columns(3)
            with r1: st.metric("推奨選出 買い目数", f"{len(bets)} 点")
            with r2: st.metric("推奨投資総額", f"{total_bet:,} 円", delta=f"資金比率: {total_bet/bankroll:.1%}")
            with r3: st.metric("最高払戻見込額", f"{int(max_ret):,} 円")
            
            res_rows = []
            for combo, amt in sorted(bets.items(), key=lambda x: x[1], reverse=True):
                p = benter_probs_dict.get(combo, 0.0)
                o = all_odds.get(combo, 0.0)
                ev = p * o
                est_ret = int(amt * o)
                res_rows.append({
                    '買い目': combo,
                    '推奨投資額 (円)': f"{amt:,} 円",
                    'Benter確率': f"{p:.2%}",
                    '実オッズ': f"{o:.1f} 倍",
                    'EV (期待値)': f"{ev:.2f}",
                    '払戻見込 (円)': f"{est_ret:,} 円"
                })
            st.dataframe(pd.DataFrame(res_rows), use_container_width=True, hide_index=True)
            
            # Supabase への保存
            try:
                db_manager.save_race_prediction(
                    race_id=race_id, race_date=date_str, venue_code=venue_code,
                    venue_name=venue_name, race_no=race_no, deadline_time="マニュアル",
                    top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=True,
                    cluster_id=cluster_id, cluster_name=cluster_name, status="investment_go"
                )
                db_manager.save_recommended_bets(race_id, bets, benter_probs_dict, all_odds)
                st.caption("💾 推論結果および推奨買い目を Supabase データベースに自動保存しました。")
            except Exception as e:
                st.caption(f"⚠️ DB保存例外: {e}")
        else:
            st.info(f"🔍 Gatekeeper は通過しましたが、厳格化条件（EV ≧ {min_ev:.2f}, Odds ≦ {max_odds:.1f}）を満たす期待値プラスの買い目が存在しないため、投資見送り（No Bet）となります。")
            try:
                db_manager.save_race_prediction(
                    race_id=race_id, race_date=date_str, venue_code=venue_code,
                    venue_name=venue_name, race_no=race_no, deadline_time="マニュアル",
                    top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=True,
                    cluster_id=cluster_id, cluster_name=cluster_name, status="no_value_bets"
                )
            except Exception: pass
