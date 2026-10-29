"""
auto_trader.py
🚤 BOATRACE AI 定量的自動通知ワーカー (Auto Trader Worker)
- 毎朝当日の全開催場・全レース締切時刻を自動取得 & スケジュール登録
- 各レースの「締切5分前」に自動トリガー実行
- 黄金ベースライン（Cluster 1除外 + Gatekeeper P1 >= 0.7438 + Extractor/Benter + Strict Optimizer）
- 投資GOサイン検知時のみ Discord Webhook へリッチ Embed 通知送信
- Supabase (PostgreSQL) への推論結果・オッズ・買い目・通知ログの自動永続化
- 開発・検証用 モックテストモード (--mock) 完備
"""

import os
import sys
import time
import datetime
import re
import json
import logging
import argparse
import itertools
from typing import Dict, List, Tuple, Optional, Any

import requests
from bs4 import BeautifulSoup
import schedule
import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# 環境変数の読み込み (.env があればロード)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DISCORD_WEBHOOK_URL = os.getenv('DISCORD_WEBHOOK_URL', '')

# 自作モジュールのインポート
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
for p in [CURRENT_DIR, PARENT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib
from odds_normalizer import probs_to_init_scores
from probability_calibration import (
    calculate_benter_probs,
    get_default_calibrator,
    get_cluster_benter_params,
    load_benter_cluster_config
)
from portfolio_optimizer import PortfolioOptimizer
import db_manager
try:
    from update_static_masters import update_all_masters
except Exception:
    update_all_masters = None

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AutoTrader")

# 定数 & パス設定
BASE_DIR = CURRENT_DIR
MODEL_HONMEI_PATH = os.path.join(BASE_DIR, 'model_honmei.txt')
MODEL_RESIDUAL_PATH = os.path.join(BASE_DIR, 'model_residual.txt')
MODEL_2ND_PATH = os.path.join(BASE_DIR, 'model_2nd.txt')
MODEL_3RD_PATH = os.path.join(BASE_DIR, 'model_3rd.txt')
DATA_DIR = os.path.join(BASE_DIR, 'app_data')
CLUSTER_1_VENUES = [2, 3, 4, 14, 22]  # 戸田02, 江戸川03, 平和島04, 鳴門14, 福岡22

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

# 会場コードマップ
VENUE_MAP: Dict[int, str] = {
    1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
    6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
    11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
    16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
    21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
}

# デフォルト設定値
DEFAULT_BANKROLL = 100000.0
DEFAULT_RISK_AVERSION = 1.0
DEFAULT_MAX_EXPOSURE = 0.05       # レース上限 5% (5,000円)
DEFAULT_MAX_CONCENTRATION = 0.02   # 買い目上限 2% (2,000円)
DEFAULT_MIN_EV = 1.25             # 最小期待値 1.25
DEFAULT_MAX_ODDS = 30.0           # 最大オッズ 30.0
DEFAULT_GATEKEEPER_TH = 0.7438    # Gatekeeper 黄金ベースライン (85th%)


# =====================================================================
# 1. スクレイパー & 特徴量生成モジュール
# =====================================================================

class BoatRaceScraper:
    _session = None

    @classmethod
    def get_session(cls) -> requests.Session:
        if cls._session is None:
            cls._session = requests.Session()
            cls._session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
                "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
                "Connection": "keep-alive"
            })
            adapter = requests.adapters.HTTPAdapter(pool_connections=15, pool_maxsize=30, max_retries=3)
            cls._session.mount("https://", adapter)
            cls._session.mount("http://", adapter)
        return cls._session

    @classmethod
    def get_soup(cls, url: str, timeout: int = 15, max_retries: int = 3) -> Optional[BeautifulSoup]:
        session = cls.get_session()
        for attempt in range(max_retries):
            try:
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                return BeautifulSoup(resp.content, 'html.parser', from_encoding='utf-8')
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1.0)
                else:
                    logger.warning(f"データ取得失敗 ({url}): {e}")
        return None


    @staticmethod
    def parse_float(text: str) -> float:
        try:
            return float(re.search(r'([\d\.]+)', text).group(1))
        except Exception:
            return 0.0

    @staticmethod
    def get_odds(date_str: str, venue_code: int, race_no: int) -> Dict[str, float]:
        """直前3連単オッズ（全120通り）をスクレイピング"""
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
    def get_race_result(date_str: str, venue_code: int, race_no: int) -> Optional[Dict[str, Any]]:
        """
        公式レース結果ページから3連単の確定着順と払戻金（100円あたり）を取得
        URL: https://www.boatrace.jp/owpc/pc/race/raceresult?rno={race_no}&jcd={jcd}&hd={date_str}
        """
        jcd = f"{int(venue_code):02d}"
        url = f"https://www.boatrace.jp/owpc/pc/race/raceresult?rno={race_no}&jcd={jcd}&hd={date_str}"
        soup = BoatRaceScraper.get_soup(url)
        if not soup:
            return None
            
        try:
            tables = soup.find_all('table')
            for t in tables:
                tbody = t.find('tbody')
                if not tbody:
                    continue
                rows = tbody.find_all('tr')
                for r in rows:
                    text = r.get_text(separator=' ', strip=True)
                    if '3連単' in text:
                        cells = [c.get_text(strip=True) for c in r.find_all(['td', 'th'])]
                        
                        # 不成立・返還・中止の検出
                        if any(w in c for c in cells for w in ['不成立', '中止', '返還', '特払い']):
                            return {
                                'combo': '不成立',
                                'payout_per_100': 100,
                                'is_cancelled': True
                            }
                            
                        combo = None
                        payout = None
                        for c in cells:
                            m_combo = re.search(r'([1-6])\s*[-=]\s*([1-6])\s*[-=]\s*([1-6])', c)
                            if m_combo:
                                combo = f"{m_combo.group(1)}-{m_combo.group(2)}-{m_combo.group(3)}"
                            if '¥' in c or '￥' in c or '円' in c:
                                try:
                                    payout = int(re.sub(r'[^\d]', '', c))
                                except Exception:
                                    pass
                        if combo and payout is not None:
                            return {
                                'combo': combo,
                                'payout_per_100': payout,
                                'is_cancelled': False
                            }
            
            # レース全体の中止・不成立検出
            page_text = soup.get_text()
            if 'レース中止' in page_text or '中止順延' in page_text or '不成立' in page_text:
                return {
                    'combo': '不成立',
                    'payout_per_100': 100,
                    'is_cancelled': True
                }
        except Exception as e:
            logger.debug(f"結果パースエラー ({url}): {e}")
        return None

    @staticmethod
    def get_race_data(date_str: str, venue_code: int, race_no: int) -> Optional[pd.DataFrame]:

        """出走表・展示タイム・気象情報を取得"""
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
            # 展示タイム & チルト パース
            for i, tb in enumerate(soup_before.select("table.is-w748 tbody")):
                tds = tb.select("td")
                ex_val = None
                tilt_val = -0.5
                if len(tds) >= 5:
                    txt = tds[4].get_text(strip=True)
                    if txt and txt != '\xa0':
                        try: ex_val = float(re.search(r'([\d\.]+)', txt).group(1))
                        except Exception: pass
                if len(tds) >= 6:
                    txt_t = tds[5].get_text(strip=True)
                    if txt_t and txt_t != '\xa0':
                        try: tilt_val = float(re.search(r'([-\+]?[\d\.]+)', txt_t).group(1))
                        except Exception: pass
                if ex_val is not None:
                    if (i+1) not in boat_before: boat_before[i+1] = {}
                    boat_before[i+1]['ex_time'] = ex_val
                    boat_before[i+1]['tilt'] = tilt_val
            
            # STパース
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
                racer_name = ''
                try: 
                    a_tag = tb.select("td")[2].select_one("a")
                    if a_tag:
                        racer_name = a_tag.get_text(strip=True)
                    txt = tb.select("td")[2].select_one("div").get_text()
                    racer_id = int(re.search(r'(\d{4})', txt).group(1))
                except Exception: pass

                branch = 'Unknown'
                weight = 52.0
                racer_rank = 'B1'
                try:
                    td2 = tb.select("td")[2]
                    txt_full = td2.get_text(" ", strip=True)
                    match_w = re.search(r'(\d{2}\.\d)kg', txt_full)
                    if match_w: weight = float(match_w.group(1))
                    prefectures = r"(群馬|埼玉|東京|福井|静岡|愛知|三重|滋賀|大阪|兵庫|徳島|香川|岡山|広島|山口|福岡|佐賀|長崎)"
                    m = re.search(prefectures, txt_full)
                    if m: branch = m.group(1)
                    m_rank = re.search(r'(A1|A2|B1|B2)', txt_full)
                    if m_rank: racer_rank = m_rank.group(1)
                except Exception: pass

                nat_win_rate = 0.0
                nat_quinella_rate = 0.0
                local_win_rate = 0.0
                try:
                    tds_all = tb.select("td")
                    if len(tds_all) > 5:
                        # 現行レイアウト: td[4]=全国(勝率 2連率 3連率), td[5]=当地(勝率 2連率 3連率)
                        nums_nat = re.findall(r'(\d+(?:\.\d+)?)', tds_all[4].get_text(" ", strip=True))
                        if len(nums_nat) >= 2:
                            nat_win_rate = float(nums_nat[0])
                            nat_quinella_rate = float(nums_nat[1])
                        elif len(nums_nat) == 1:
                            nat_win_rate = float(nums_nat[0])
                        nums_loc = re.findall(r'(\d+(?:\.\d+)?)', tds_all[5].get_text(" ", strip=True))
                        if len(nums_loc) >= 1:
                            local_win_rate = float(nums_loc[0])
                    if nat_win_rate == 0.0 and len(tds_all) > 3:
                        # 旧レイアウト互換フォールバック
                        col3_txt = tds_all[3].get_text(" ", strip=True)
                        clean_txt = re.sub(r'[FLK]\d+', '', col3_txt) 
                        nums = re.findall(r'(\d+(?:\.\d+)?)', clean_txt)
                        if len(nums) >= 5:
                            nat_win_rate = float(nums[1])
                            nat_quinella_rate = float(nums[2])
                            local_win_rate = float(nums[3])
                        elif len(nums) >= 4:
                            nat_win_rate = float(nums[0])
                            nat_quinella_rate = float(nums[1])
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
                    'racer_name': racer_name,
                    'motor_rate': motor,
                    'boat_rate': boat,
                    'exhibition_time': boat_before[bn]['ex_time'],
                    'exhibition_start_timing': boat_before.get(bn, {}).get('st', 0.20),
                    'pred_course': boat_before.get(bn, {}).get('pred_course', bn),
                    'tilt': boat_before.get(bn, {}).get('tilt', -0.5),
                    'wind_direction': weather['wind_direction'],
                    'wind_speed': weather['wind_speed'],
                    'wave_height': weather['wave_height'],
                    'prior_results': prior_results,
                    'branch': branch,
                    'weight': weight,
                    'racer_rank': racer_rank,
                    'nat_win_rate': nat_win_rate,
                    'nat_quinella_rate': nat_quinella_rate,
                    'local_win_rate': local_win_rate
                }

                rows.append(row)
        except Exception as e:
            logger.error(f"出走表パースエラー: {e}")
            return None
            
        return pd.DataFrame(rows)


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    if 'prior_results' in df.columns:
        df['is_F_holder'] = df['prior_results'].astype(str).apply(lambda x: 1 if 'F' in x else 0)
    else:
        df['is_F_holder'] = 0
        
    st_col = 'course_avg_st' if 'course_avg_st' in df.columns else 'exhibition_start_timing'
    if st_col in df.columns:
        df['corrected_st'] = df[st_col] + (df['is_F_holder'] * 0.05)
    else:
        df['corrected_st'] = 0.20
        
    if 'race_id' not in df.columns:
        df['race_id'] = 'single_race'
    sort_col = 'pred_course' if 'pred_course' in df.columns else 'boat_number'
    df = df.sort_values(['race_id', sort_col])

    prev_sts = df.groupby('race_id')['corrected_st'].shift(1)
    df['inner_st_gap_corrected'] = (df['corrected_st'] - prev_sts).fillna(0.0)
    df.loc[df[sort_col] == 1, 'inner_st_gap_corrected'] = 0.0
    
    if 'motor_rate' in df.columns and 'exhibition_time' in df.columns:
        df['motor_rank'] = df.groupby('race_id')['motor_rate'].rank(ascending=False, method='min')
        df['tenji_rank'] = df.groupby('race_id')['exhibition_time'].rank(ascending=True, method='min')
        df['motor_gap'] = df['motor_rank'] - df['tenji_rank']
    else:
        df['motor_gap'] = 0.0
        
    if 'course_quinella_rate' in df.columns and 'nat_quinella_rate' in df.columns:
        df['specialist_score'] = (df['course_quinella_rate'] - df['nat_quinella_rate']).fillna(0.0)
    elif 'course_1st_rate' in df.columns and 'nat_win_rate' in df.columns:
        df['specialist_score'] = df['course_1st_rate'] - df['nat_win_rate']
    elif 'venue_course_1st_rate' in df.columns and 'nat_win_rate' in df.columns:
        df['specialist_score'] = df['venue_course_1st_rate'] - df['nat_win_rate']
    else:
        df['specialist_score'] = 0.0
        
    if 'nige_count' in df.columns and 'course_run_count' in df.columns:
        df['my_nige_rate'] = df['nige_count'] / (df['course_run_count'] + 1.0)
        df['my_sashi_rate'] = df['sashi_count'] / (df['course_run_count'] + 1.0)
        df['my_makuri_rate'] = df['makuri_count'] / (df['course_run_count'] + 1.0)
        
        c1_nige_rate = df.groupby('race_id')['my_nige_rate'].transform('first')
        df['sashi_potential'] = (df['my_sashi_rate'] * (1.0 - c1_nige_rate)).fillna(0.0)
        df.loc[df[sort_col] == 1, 'sashi_potential'] = 0.0
        
        df['st_rank'] = df.groupby('race_id')['corrected_st'].rank(ascending=True)
        inner_st_rank = df.groupby('race_id')['st_rank'].shift(1).fillna(3.5)
        df['makuri_potential'] = (df['my_makuri_rate'] * inner_st_rank).fillna(0.0)
        df.loc[df[sort_col] == 1, 'makuri_potential'] = 0.0
    else:
        df['sashi_potential'] = 0.0
        df['makuri_potential'] = 0.0
    df = df.sort_values(['race_id', 'boat_number'])
        
    bias_path = os.path.join(DATA_DIR, 'venue_frame_bias.csv')
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

    # 級別関連特徴量 (racer_rank_num, rank_numeric, level_adjusted_win_rate, rank_skill_gap)
    rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
    if 'racer_rank' in df.columns:
        df['racer_rank_num'] = df['racer_rank'].astype(str).str.strip().map(rank_map).fillna(2).astype(int)
    elif 'racer_rank_num' not in df.columns:
        df['racer_rank_num'] = 2
    df['rank_numeric'] = df['racer_rank_num']

    rank_mean_lookup = {4: 6.6695, 3: 5.7890, 2: 4.5525, 1: 2.6491}
    if 'nat_win_rate' in df.columns:
        df['level_adjusted_win_rate'] = df['nat_win_rate'] * df['rank_numeric']
        expected_rank_mean = df['rank_numeric'].map(rank_mean_lookup).fillna(4.5525)
        df['rank_skill_gap'] = (df['nat_win_rate'] - expected_rank_mean).fillna(0.0)
    else:
        df['level_adjusted_win_rate'] = 0.0
        df['rank_skill_gap'] = 0.0
        
    return df


def fetch_series_momentum(venue_code: int, racer_ids: list, race_date: str = None, current_times: dict = None) -> dict:
    """
    対象会場・今節（同一会場で過去7日以内〜当日）における出走選手の過去展示タイム履歴をDBから取得し、
    前走展示タイム差 (ex_momentum_diff) および 節間平均偏差 (ex_momentum_deviation) を算出する。
    リアルタイムスクレイピング時（未確定・DB未保存データ）でも current_times とDB履歴を自動結合して計算。
    """
    momentum_dict = {rid: {'ex_momentum_diff': 0.0, 'ex_momentum_deviation': 0.0} for rid in racer_ids}
    if not racer_ids:
        return momentum_dict

    sqlite_paths = [os.path.join(BASE_DIR, 'boatrace.db'), r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db']
    sqlite_db = None
    for sp in sqlite_paths:
        if os.path.exists(sp):
            sqlite_db = sp
            break

    racer_times = {int(rid): [] for rid in racer_ids}
    found_in_sqlite = False

    if sqlite_db:
        try:
            import sqlite3
            conn = sqlite3.connect(sqlite_db)
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(racer_ids))
            
            if race_date:
                # 8桁の YYYYMMDD 形式なら SQLite用に YYYY-MM-DD に変換
                if len(str(race_date)) == 8 and str(race_date).isdigit():
                    formatted_date = f"{str(race_date)[:4]}-{str(race_date)[4:6]}-{str(race_date)[6:]}"
                else:
                    formatted_date = str(race_date)
                    
                date_filter = "AND r.race_date >= date(?, '-7 days') AND r.race_date <= ?"
                params = [int(venue_code)] + [int(r) for r in racer_ids] + [formatted_date, formatted_date]
            else:
                date_filter = "AND r.race_date >= date('now', '-7 days')"
                params = [int(venue_code)] + [int(r) for r in racer_ids]

            query = f"""
            SELECT re.racer_id, bi.exhibition_time
            FROM before_info bi
            JOIN races r ON bi.race_id = r.race_id
            JOIN race_entries re ON bi.race_id = re.race_id AND bi.boat_number = re.boat_number
            WHERE CAST(r.venue_code AS INTEGER) = ?
              AND re.racer_id IN ({placeholders})
              {date_filter}
              AND bi.exhibition_time > 0
            ORDER BY r.race_date ASC, r.race_number ASC
            """

            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            if rows:
                for r_id, ex_time in rows:
                    r_id = int(r_id)
                    if r_id in racer_times:
                        racer_times[r_id].append(float(ex_time))
                found_in_sqlite = True
        except Exception as e:
            logger.debug(f"fetch_series_momentum SQLite error: {e}")

    # SQLiteで見つからないか0件の場合は Supabase (クラウド) にフォールバック
    total_historical_runs = sum(len(v) for v in racer_times.values())
    if not found_in_sqlite or total_historical_runs == 0:
        try:
            cloud_times_map = db_manager.get_cloud_series_momentum(venue_code, racer_ids, race_date=race_date)
            if cloud_times_map:
                for rid, t_list in cloud_times_map.items():
                    if t_list:
                        racer_times[int(rid)] = t_list
        except Exception as e:
            logger.debug(f"fetch_series_momentum Supabase fallback error: {e}")

    for r_id in racer_ids:
        r_id = int(r_id)
        times = racer_times.get(r_id, [])
        cur_ex = float(current_times.get(r_id, 0.0)) if current_times and pd.notna(current_times.get(r_id)) else 0.0
        
        if cur_ex > 0:
            if len(times) == 0:
                momentum_dict[r_id]['ex_momentum_diff'] = 0.0
                momentum_dict[r_id]['ex_momentum_deviation'] = 0.0
            elif len(times) >= 1 and abs(times[-1] - cur_ex) < 1e-4:
                # 過去データとして既にDBに含まれる場合
                if len(times) >= 2:
                    current_ex_val = times[-1]
                    prev_ex_val = times[-2]
                    exp_mean = np.mean(times)
                    momentum_dict[r_id]['ex_momentum_diff'] = float(current_ex_val - prev_ex_val)
                    momentum_dict[r_id]['ex_momentum_deviation'] = float(current_ex_val - exp_mean)
                else:
                    momentum_dict[r_id]['ex_momentum_diff'] = 0.0
                    momentum_dict[r_id]['ex_momentum_deviation'] = 0.0
            else:
                # リアルタイム未保存レース（DBの過去走 + 今回スクレイピングした展示タイムを結合）
                prev_ex_val = times[-1]
                all_times = times + [cur_ex]
                exp_mean = np.mean(all_times)
                momentum_dict[r_id]['ex_momentum_diff'] = float(cur_ex - prev_ex_val)
                momentum_dict[r_id]['ex_momentum_deviation'] = float(cur_ex - exp_mean)
        else:
            # DB内の既存データからのみ算出
            if len(times) >= 2:
                current_ex_val = times[-1]
                prev_ex_val = times[-2]
                exp_mean = np.mean(times)
                momentum_dict[r_id]['ex_momentum_diff'] = float(current_ex_val - prev_ex_val)
                momentum_dict[r_id]['ex_momentum_deviation'] = float(current_ex_val - exp_mean)
            else:
                momentum_dict[r_id]['ex_momentum_diff'] = 0.0
                momentum_dict[r_id]['ex_momentum_deviation'] = 0.0

        
    return momentum_dict




class FeatureEngineer:
    @staticmethod
    def process(df: pd.DataFrame, venue_name: str, race_date: str = None) -> pd.DataFrame:
        df['venue_name'] = venue_name
        
        # 1. 選手IDと進入コースの安全な型変換（NaN/Noneによる例外を防止）
        df['racer_id'] = pd.to_numeric(df.get('racer_id', 9999), errors='coerce').fillna(9999).astype(int)
        if 'pred_course' in df.columns:
            df['pred_course'] = pd.to_numeric(df['pred_course'], errors='coerce').fillna(df.get('boat_number', 1)).astype(int)
        else:
            df['pred_course'] = pd.to_numeric(df.get('boat_number', 1), errors='coerce').fillna(1).astype(int)

        try:
            r_course = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_course.csv'))
            r_venue = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_venue.csv'))
            v_course = pd.read_csv(os.path.join(DATA_DIR, 'static_venue_course.csv'))
            r_params = pd.read_csv(os.path.join(DATA_DIR, 'static_racer_params.csv'))
            
            r_course['RacerID'] = pd.to_numeric(r_course['RacerID'], errors='coerce').fillna(0).astype(int)
            r_course['Course'] = pd.to_numeric(r_course['Course'], errors='coerce').fillna(0).astype(int)
            r_venue['RacerID'] = pd.to_numeric(r_venue['RacerID'], errors='coerce').fillna(0).astype(int)
            v_course['course_number'] = pd.to_numeric(v_course['course_number'], errors='coerce').fillna(0).astype(int)
            r_params['racer_id'] = pd.to_numeric(r_params['racer_id'], errors='coerce').fillna(0).astype(int)

            df = df.merge(r_course, left_on=['racer_id', 'pred_course'], right_on=['RacerID', 'Course'], how='left')
            df.rename(columns={
                'RacesRun': 'course_run_count',
                'QuinellaRate': 'course_quinella_rate',
                'TrifectaRate': 'course_trifecta_rate',
                'FirstPlaceRate': 'course_1st_rate',
                'AvgStartTiming': 'course_avg_st',
                'Nige': 'nige_count', 
                'Makuri': 'makuri_count',
                'Sashi': 'sashi_count',
                'Makurizashi': 'makurizashi_count'
            }, inplace=True)

            venue_map_rev = {
                '桐生': 1, '戸田': 2, '江戸川': 3, '平和島': 4, '多摩川': 5,
                '浜名湖': 6, '蒲郡': 7, '常滑': 8, '津': 9, '三国': 10,
                'びわこ': 11, '住之江': 12, '尼崎': 13, '鳴門': 14, '丸亀': 15,
                '児島': 16, '宮島': 17, '徳山': 18, '下関': 19, '若松': 20,
                '芦屋': 21, '福岡': 22, '唐津': 23, '大村': 24
            }
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

            v_course_clean = v_course.rename(columns={'venue_code': 'vc_join'})
            df = df.merge(v_course_clean, left_on=['venue_code_int', 'pred_course'], right_on=['vc_join', 'course_number'], how='left')
            df.drop(columns=['vc_join'], inplace=True, errors='ignore')
            df['venue_code_y'] = df['venue_code_int'].astype(str).str.zfill(2)
            df.rename(columns={'rate_1st': 'venue_course_1st_rate', 'rate_2nd': 'venue_course_2nd_rate', 'rate_3rd': 'venue_course_3rd_rate'}, inplace=True)
            
            # r_params からは st_std_dev のみをマージし、nat_win_rate / nige_count 等の列名衝突 (_x, _y) を完全に防止
            r_params_clean = r_params[['racer_id', 'st_std_dev']].drop_duplicates(subset=['racer_id']) if 'st_std_dev' in r_params.columns else r_params[['racer_id']].drop_duplicates()
            df = df.merge(r_params_clean, on='racer_id', how='left')
            if 'nat_win_rate' in r_params.columns:
                nw_map = dict(zip(r_params['racer_id'], r_params['nat_win_rate']))
                if 'nat_win_rate' in df.columns:
                    df['nat_win_rate'] = pd.to_numeric(df['nat_win_rate'], errors='coerce').replace(0.0, np.nan).fillna(df['racer_id'].map(nw_map)).fillna(0.0)
                else:
                    df['nat_win_rate'] = df['racer_id'].map(nw_map).fillna(0.0)
        except Exception as e:
            logger.warning(f"⚠️ 静的マスタ特徴量マージ例外 (デフォルト値で補完します): {e}")
        
        # 2. 静的特徴量の確実なデフォルト値補完 (マージ失敗時や未登録選手対策)
        default_feature_cols = {
            'makuri_count': 0.0,
            'nige_count': 0.0,
            'sashi_count': 0.0,
            'makurizashi_count': 0.0,
            'nat_win_rate': 0.0,
            'nat_quinella_rate': 0.0,
            'st_std_dev': 0.05,
            'course_run_count': 0.0,
            'local_win_rate': 0.0,
            'course_quinella_rate': 0.0,
            'course_trifecta_rate': 0.0,
            'course_1st_rate': 0.0,
            'course_avg_st': 0.20,
            'venue_course_1st_rate': 0.0,
            'venue_course_2nd_rate': 0.0,
            'venue_course_3rd_rate': 0.0,
            'wall_strength': 0.0,
            'follow_potential': 0.0
        }
        for c, def_val in default_feature_cols.items():
            if c not in df.columns:
                df[c] = def_val
            else:
                df[c] = df[c].fillna(def_val)
            
        def parse_prior(x):
            if isinstance(x, (int, float)) and not np.isnan(x):
                x = str(int(x))
            if not isinstance(x, str): return 3.5
            try:
                ranks = [float(c) for c in x if c in '123456']
                if ranks: return sum(ranks) / len(ranks)
            except Exception: pass
            return 3.5
            
        df['series_avg_rank'] = df['prior_results'].apply(parse_prior) if 'prior_results' in df.columns else 3.5
        denom = df['course_run_count'].replace(0, 1)
        df['makuri_rate'] = df['makuri_count'] / denom
        df['nige_rate'] = df['nige_count'] / denom
        df['sashi_rate'] = df.get('sashi_count', 0.0) / denom
        df['makurizashi_rate'] = df.get('makurizashi_count', 0.0) / denom

        # Advanced Features
        df = add_advanced_features(df)
        df = df.sort_values('pred_course')
        
        # 学習時 (make_data_set.py) と完全一致: inner_st_gap と slit_formation は exhibition_start_timing から算出
        ex_st_s = pd.to_numeric(df['exhibition_start_timing'], errors='coerce').fillna(0.15) if 'exhibition_start_timing' in df.columns else pd.Series(0.15, index=df.index)
        inner_st_s = ex_st_s.shift(1)
        df['inner_st_gap'] = (ex_st_s - inner_st_s).fillna(0.0)
        outer_st_s = ex_st_s.shift(-1)
        avg_neighbor = (inner_st_s.fillna(ex_st_s) + outer_st_s.fillna(ex_st_s)) / 2.0
        df['slit_formation'] = (ex_st_s - avg_neighbor).fillna(0.0)

        c1_nige = df.loc[df['pred_course']==1, 'nige_rate']
        val = c1_nige.values[0] if len(c1_nige) > 0 else 0.5
        df['anti_nige_potential'] = df['makuri_rate'] * (1 - val)
        
        # 安全な wall_strength & follow_potential 算出
        quinella_s = df['course_quinella_rate'].fillna(0.0)
        df['wall_strength'] = quinella_s.shift(1).fillna(0.0)
        df['follow_potential'] = df['makuri_rate'].shift(1).fillna(0.0) * quinella_s
        
        min_t = df['exhibition_time'].min() if 'exhibition_time' in df.columns else 6.8
        if pd.isna(min_t): min_t = 6.8
        mean_t = df['exhibition_time'].mean() if 'exhibition_time' in df.columns else 6.8
        if pd.isna(mean_t): mean_t = 6.8
        std_t = df['exhibition_time'].std() if 'exhibition_time' in df.columns else 1.0
        if std_t == 0 or np.isnan(std_t): std_t = 1.0
        
        ex_time_s = df['exhibition_time'].fillna(mean_t) if 'exhibition_time' in df.columns else pd.Series(mean_t, index=df.index)
        df['tenji_z_score'] = (mean_t - ex_time_s) / std_t
        df['linear_rank'] = ex_time_s.rank(method='min', ascending=True)
        df['is_linear_leader'] = (df['linear_rank'] == 1).astype(int)

        # 新規代替モメンタム & レース内展示偏差
        df['ex_diff_from_race_min'] = (ex_time_s - min_t).fillna(0.0)
        df['ex_diff_from_race_mean'] = (ex_time_s - mean_t).fillna(0.0)
        df['ex_rank_in_race'] = df['linear_rank']
        
        # 節間展示タイムモメンタム（動的DBクエリ結果をマッピング）
        racer_ids_list = df['racer_id'].dropna().astype(int).tolist()
        v_code_val = int(df['venue_code_int'].iloc[0]) if 'venue_code_int' in df.columns and len(df) > 0 else 1
        current_times = dict(zip(df['racer_id'].astype(int), pd.to_numeric(df['exhibition_time'], errors='coerce'))) if 'exhibition_time' in df.columns else None
        momentum_map = fetch_series_momentum(v_code_val, racer_ids_list, race_date, current_times=current_times)
        
        df['ex_momentum_diff'] = df['racer_id'].map(lambda r: momentum_map.get(int(r), {}).get('ex_momentum_diff', 0.0) if pd.notna(r) else 0.0).fillna(0.0)
        df['ex_momentum_deviation'] = df['racer_id'].map(lambda r: momentum_map.get(int(r), {}).get('ex_momentum_deviation', 0.0) if pd.notna(r) else 0.0).fillna(0.0)

        
        if 'weight_x' in df.columns: df['weight'] = df['weight_x']
        if 'weight' not in df.columns: df['weight'] = 52.0
        df['weight_diff'] = df['weight'] - df['weight'].mean()
        df['high_wind_alert'] = (df['wind_speed'] >= 5).astype(int)

        # 新規環境クロス (風速・波高)
        df['is_strong_wind'] = (df['wind_speed'] >= 4.0).astype(float)
        df['is_gale_wind'] = (df['wind_speed'] >= 6.0).astype(float)
        df['wind_makuri_cross'] = df['wind_speed'] * df['makuri_rate']
        df['strong_wind_makuri'] = df['is_strong_wind'] * df['makuri_rate']
        df['wind_makurizashi_cross'] = df['wind_speed'] * df['makurizashi_rate']
        df['strong_wind_outer_adv'] = df['is_strong_wind'] * (df['pred_course'] >= 3).astype(float)
        df['wind_nige_vulnerability'] = df['wind_speed'] * (1.0 - df['nige_rate']) * (df['pred_course'] == 1).astype(float)


        df['wave_weight_prod'] = df['wave_height'] * df['weight']
        df['wave_weight_ratio'] = df['wave_height'] / np.maximum(df['weight'], 40.0)
        df['is_high_wave'] = (df['wave_height'] >= 4.0).astype(float)
        df['high_wave_heavy_penalty'] = df['is_high_wave'] * np.maximum(0.0, df['weight'] - 52.0)
        df['high_wave_inner_risk'] = df['is_high_wave'] * (df['pred_course'] == 1).astype(float)
        
        df['nat_win_rate'] = pd.to_numeric(df['nat_win_rate'], errors='coerce').fillna(0.0)
        df['local_win_rate'] = pd.to_numeric(df['local_win_rate'], errors='coerce').fillna(0.0)
        df['local_perf_diff'] = df['local_win_rate'] - df['nat_win_rate']

        wind_dir_to_angle = {
            '北': 0.0, '北北東': 22.5, '北東': 45.0, '東北東': 67.5, '東': 90.0, '東南東': 112.5,
            '南東': 135.0, '南南東': 157.5, '南': 180.0, '南南西': 202.5, '南西': 225.0, '西南西': 247.5,
            '西': 270.0, '西北西': 292.5, '北西': 315.0, '北北西': 337.5
        }
        def wind_deg_from_val(x):
            try:
                if isinstance(x, (int, float)) and not np.isnan(x):
                    ix = int(x)
                    return (ix - 1) * 22.5 if 1 <= ix <= 16 else 0.0
                if isinstance(x, str):
                    if x.isdigit():
                        ix = int(x)
                        return (ix - 1) * 22.5 if 1 <= ix <= 16 else 0.0
                    return wind_dir_to_angle.get(x, 0.0)
            except Exception:
                pass
            return 0.0

        df['wind_angle_deg'] = df['wind_direction'].apply(wind_deg_from_val)
        venue_tailwind_from = {
            '桐生': 135, '戸田': 90, '江戸川': 180, '平和島': 180, '多摩川': 270,
            '浜名湖': 180, '蒲郡': 270, '常滑': 270, '津': 135, '三国': 180,
            'びわこ': 225, '住之江': 270, '尼崎': 90, '鳴門': 135, '丸亀': 180,
            '児島': 225, '宮島': 270, '徳山': 135, '下関': 270, '若松': 270,
            '芦屋': 135, '福岡': 0, '唐津': 135, '大村': 315
        }
        df['venue_tailwind_deg'] = df['venue_name'].map(venue_tailwind_from).fillna(0)
        angle_diff_rad = np.radians(df['wind_angle_deg'] - df['venue_tailwind_deg'])
        df['wind_vector_long'] = df['wind_speed'] * np.cos(angle_diff_rad)
        df['wind_vector_lat'] = df['wind_speed'] * np.sin(angle_diff_rad)

        if 'race_date' not in df.columns: df['race_date'] = '20000101'
        
        # LightGBM の pandas_categorical (['北', '北東', '北西', '南', '南東', '南西', '東', '無風', '西']) に完全適合させる
        wind_map_8 = {
            0: '無風', 17: '無風', '0': '無風', '17': '無風',
            1: '北', 2: '北東', 3: '北東', 4: '東', 5: '東', 6: '南東', 7: '南東', 8: '南',
            9: '南', 10: '南西', 11: '南西', 12: '西', 13: '西', 14: '北西', 15: '北西', 16: '北',
            '北北東': '北東', '東北東': '東', '東南東': '南東', '南南東': '南',
            '南南西': '南西', '西南西': '西', '西北西': '北西', '北北西': '北'
        }
        df['wind_direction'] = df['wind_direction'].replace(wind_map_8).astype(str).replace('nan', '無風')

        df = df.loc[:, ~df.columns.duplicated()]
        df = df.sort_values('boat_number').reset_index(drop=True)

        for col in df.columns:
            if col not in ['race_id', 'race_date', 'venue_name', 'prior_results', 'wind_direction', 'branch', 'class', 'racer_class', 'venue_code_y']:
                df[col] = pd.to_numeric(df[col], errors='coerce')


        return df.copy()


def prepare_features_for_model(df_feat: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
    """LightGBM Booster の pandas_categorical 仕様に適合した DataFrame を構築"""
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
                raw_val = df_feat[f]
                if isinstance(raw_val, pd.DataFrame):
                    val_series = raw_val.iloc[:, 0].astype(str)
                else:
                    val_series = raw_val.astype(str)
            elif f == 'venue_code_y' and 'venue_code_int' in df_feat.columns:
                val_series = df_feat['venue_code_int'].astype(str).str.zfill(2)
            elif f == 'venue_code_y' and 'temp_venue_code' in df_feat.columns:
                val_series = df_feat['temp_venue_code'].astype(str).str.zfill(2)
            else:
                val_series = pd.Series([cat_list[0]] * len(df_feat), index=df_feat.index)
            df_out[f] = pd.Categorical(val_series, categories=cat_list)
        else:
            if f in df_feat.columns:
                raw_val = df_feat[f]
                if isinstance(raw_val, pd.DataFrame):
                    raw_val = raw_val.iloc[:, 0]
                df_out[f] = pd.to_numeric(raw_val, errors='coerce').fillna(0.0).astype(float)
            elif f == 'syn_win_rate':
                df_out[f] = 0.0
            else:
                df_out[f] = 0.0
                
    return df_out



# =====================================================================
# 2. スケジュール取得 & 管理モジュール
# =====================================================================

def fetch_today_venues(date_str: str) -> List[Tuple[int, str]]:
    """当日の開催場一覧を取得"""
    url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={date_str}"
    venues = []
    try:
        soup = BoatRaceScraper.get_soup(url, timeout=20)
        if soup:
            for a in soup.find_all('a', href=re.compile(r'/owpc/pc/race/raceindex\?jcd=(\d+)')):
                m = re.search(r'jcd=(\d+)', a['href'])
                if m:
                    jcd = int(m.group(1))
                    vname = VENUE_MAP.get(jcd, f"場{jcd:02d}")
                    if (jcd, vname) not in venues:
                        venues.append((jcd, vname))
    except Exception as e:
        logger.error(f"開催会場一覧の取得に失敗しました: {e}")
    return sorted(venues, key=lambda x: x[0])


def fetch_venue_race_deadlines(date_str: str, venue_code: int) -> List[Dict[str, Any]]:
    """指定会場の全レース締切時刻を取得"""
    url = f"https://www.boatrace.jp/owpc/pc/race/raceindex?jcd={venue_code:02d}&hd={date_str}"
    races = []
    try:
        soup = BoatRaceScraper.get_soup(url, timeout=20)
        if soup:
            for tr in soup.select('table tbody tr'):
                tds = tr.select('th, td')
                if len(tds) >= 2:
                    r_txt = tds[0].get_text(strip=True)
                    d_txt = tds[1].get_text(strip=True)
                    r_match = re.match(r'(\d+)R', r_txt)
                    d_match = re.match(r'(\d{1,2}:\d{2})', d_txt)
                    if r_match and d_match:
                        rno = int(r_match.group(1))
                        dtime_str = d_match.group(1)
                        if len(dtime_str) == 4:  # "9:30" -> "09:30"
                            dtime_str = "0" + dtime_str
                        races.append({
                            'race_no': rno,
                            'deadline_str': dtime_str
                        })
    except Exception as e:
        logger.error(f"[{venue_code:02d}] レース締切時刻の取得に失敗しました: {e}")
    return sorted(races, key=lambda x: x['race_no'])


# =====================================================================
# 3. Discord Webhook 通知モジュール
# =====================================================================

def send_discord_notification(
    webhook_url: str,
    date_str: str,
    venue_code: int,
    venue_name: str,
    race_no: int,
    deadline_str: str,
    top_boat: int,
    max_p1: float,
    prob_gap: float,
    cluster_id: int,
    cluster_name: str,
    bets: Dict[str, int],
    benter_probs: Dict[str, float],
    all_odds: Dict[str, float],
    bankroll: float,
    dry_run: bool = False
) -> bool:
    """投資GOサインが点灯したレースを Discord Webhook へ Embed 送信"""
    total_bet = sum(bets.values())
    max_return = max([amt * all_odds.get(c, 0.0) for c, amt in bets.items()]) if bets else 0
    race_id = f"{date_str}_{venue_code}_{race_no}"
    
    # 買い目テーブル整形
    table_lines = [
        "```",
        f"{'買い目':^7} | {'推奨金額':^8} | {'実オッズ':^7} | {'EV':^6} | {'払戻見込':^9}",
        "-" * 50
    ]
    for combo, amt in sorted(bets.items(), key=lambda x: x[1], reverse=True):
        p = benter_probs.get(combo, 0.0)
        o = all_odds.get(combo, 0.0)
        ev = p * o
        est_ret = int(amt * o)
        table_lines.append(f"{combo:^7} | {amt:>6,d}円 | {o:>5.1f}倍 | {ev:>5.2f} | {est_ret:>7,d}円")
    table_lines.append("```")
    bets_formatted_table = "\n".join(table_lines)
    
    title_text = f"🚀 【投資GOサイン】{venue_name} {race_no}R（締切 {deadline_str} / 発走5分前）"
    
    embed = {
        "title": title_text,
        "description": (
            "**全関門突破！** 水面適格 × Gatekeeper通過 × EV残差エッジ検知\n"
            "Markowitz / SLSQP 最適化に基づく推奨ポートフォリオ資金配分です。"
        ),
        "color": 0x00E676,  # エメラルドグリーン
        "fields": [
            {
                "name": "🛡️ Gatekeeper 信頼度",
                "value": f"**P1 = {max_p1:.1%}** ({top_boat}号艇本命 / 2位差: {prob_gap:+.1%})",
                "inline": True
            },
            {
                "name": "🏟️ 会場クラスタ",
                "value": f"**{cluster_name}** (Cluster {cluster_id})",
                "inline": True
            },
            {
                "name": "💰 推奨投資総額",
                "value": f"**{total_bet:,} 円** (資金比率: {total_bet/bankroll:.1%})",
                "inline": True
            },
            {
                "name": f"🎯 推奨買い目リスト ({len(bets)}点 / 最高払戻 {int(max_return):,}円)",
                "value": bets_formatted_table,
                "inline": False
            }
        ],
        "footer": {
            "text": f"BOATRACE AI Dual Quant System (Golden Baseline) • {date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }
    
    logger.info(f"✨ [DISCORD NOTIFICATION TRIGGERED] {venue_name} {race_no}R | 投資総額: {total_bet:,}円 | 買い目: {len(bets)}点")
    
    # Supabase へ通知ログを保存
    try:
        db_manager.save_notification_log(
            race_id=race_id,
            channel='discord',
            title=title_text,
            payload=embed,
            status='dry_run' if (dry_run or not webhook_url) else 'sent'
        )
    except Exception as e:
        logger.warning(f"通知ログ保存エラー: {e}")
    
    if dry_run or not webhook_url:
        logger.info("[DRY-RUN / NO WEBHOOK] Discord送信をスキップしました (Payloadはコンソールに出力):")
        print("\n" + "=" * 65)
        print(f"🚀 【投資GOサイン】{venue_name} {race_no}R (締切 {deadline_str})")
        print(f"Gatekeeper P1: {max_p1:.1%} ({top_boat}号艇) | クラスタ: {cluster_name}")
        print(f"推奨投資総額: {total_bet:,}円 (最高払戻: {int(max_return):,}円)")
        print(bets_formatted_table)
        print("=" * 65 + "\n")
        return True
        
    try:
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        if resp.status_code in (200, 204):
            logger.info("✅ Discord Webhook 送信成功！")
            return True
        else:
            logger.warning(f"⚠️ Discord Webhook 送信失敗 (Status: {resp.status_code}): {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ Discord Webhook 送信中にエラーが発生しました: {e}")
        return False


# =====================================================================
# 3.5 万舟スナイパー (💣 万舟スナイパー) 通知 & 算出モジュール
# =====================================================================

def send_manshu_discord_notification(
    webhook_url: str,
    date_str: str,
    venue_code: int,
    venue_name: str,
    race_no: int,
    deadline_str: str,
    is_tier1: bool,
    b1_drop: float,
    outer_ov: float,
    outer_boat: int,
    bets: Dict[str, int],
    manshu_probs: Dict[str, float],
    all_odds: Dict[str, float],
    dry_run: bool = False
) -> bool:
    """万舟スナイパー (💣 万舟スナイパー) 発動レースを Discord Webhook へ専用 Embed 送信"""
    if not bets:
        return False
    total_bet = sum(bets.values())
    returns = [int((amt / 100.0) * all_odds.get(c, 0.0) * 100.0) for c, amt in bets.items()]
    min_return = min(returns) if returns else 0
    max_return = max(returns) if returns else 0
    race_id = f"{date_str}_{venue_code}_{race_no}"

    table_lines = [
        "```",
        f"{'買い目':^7} | {'金額':^6} | {'実オッズ':^8} | {'EV':^5} | {'的中払戻':^9}",
        "-" * 50
    ]
    for combo, amt in sorted(bets.items(), key=lambda x: manshu_probs.get(x[0], 0.0), reverse=True):
        p = manshu_probs.get(combo, 0.0)
        o = all_odds.get(combo, 0.0)
        ev = p * o
        est_ret = int((amt / 100.0) * o * 100.0)
        table_lines.append(f"{combo:^7} | {amt:>4,d}円 | {o:>6.1f}倍 | {ev:>5.2f} | {est_ret:>7,d}円")
    table_lines.append("```")
    bets_formatted_table = "\n".join(table_lines)

    if is_tier1:
        tier_badge = "🔥 【激熱万舟スナイパー発動 (Tier 1)】"
        tier_desc = (
            "**イン過信危険水域 × 外枠急浮上検知！** (2026年OOT検証: **万舟的中率 10.5% / 回収率 443.5%**)\n"
            "市場が1号艇を過大評価し、4〜6号艇の台頭を見落としている高期待値・少額万舟スナイプレースです。"
        )
        embed_color = 0xFF3D00  # クリムゾンオレンジ
    else:
        tier_badge = "💣 【万舟スナイパー好機 (Tier 2)】"
        tier_desc = (
            "**イン過信 × 外枠穴オーバーレイ検知！** (2026年OOT検証: **万舟的中率 5.9% / 回収率 243.2%**)\n"
            "オッズ100〜500倍帯にEV≥1.20の歪みが発生している少額100円フラット万舟スナイプレースです。"
        )
        embed_color = 0xAA00FF  # ネオンパープル

    title_text = f"{tier_badge} {venue_name} {race_no}R（締切 {deadline_str}）"

    embed = {
        "title": title_text,
        "description": tier_desc,
        "color": embed_color,
        "fields": [
            {
                "name": "📉 1号艇・市場過信度",
                "value": f"**+{b1_drop:.1%}** (市場勝率 − AI残差)",
                "inline": True
            },
            {
                "name": "🚀 外枠(4-6号艇) AI激推し",
                "value": f"**{outer_boat}号艇 (+{outer_ov:.1%})**",
                "inline": True
            },
            {
                "name": "💰 推奨投資額 (100円均等)",
                "value": f"**{total_bet:,} 円** ({len(bets)}点 × 100円)",
                "inline": True
            },
            {
                "name": f"💣 万舟スナイプ買い目 ({len(bets)}点 / 払戻レンジ {min_return:,}円〜{max_return:,}円)",
                "value": bets_formatted_table,
                "inline": False
            }
        ],
        "footer": {
            "text": f"BOATRACE AI Man-shu Sniper (100x-500x) • {date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"
        },
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

    logger.info(f"💣 [MANSHU DISCORD TRIGGERED] {venue_name} {race_no}R | {'Tier1(激熱)' if is_tier1 else 'Tier2(好機)'} | 投資: {total_bet:,}円 ({len(bets)}点)")

    try:
        db_manager.save_notification_log(
            race_id=race_id,
            channel='discord_manshu',
            title=title_text,
            payload=embed,
            status='dry_run' if (dry_run or not webhook_url) else 'sent'
        )
    except Exception as e:
        logger.warning(f"万舟通知ログ保存エラー: {e}")

    if dry_run or not webhook_url:
        logger.info("[DRY-RUN / NO WEBHOOK] 万舟スナイパーDiscord送信をスキップしました:")
        print("\n" + "=" * 65)
        print(title_text)
        print(f"1号艇過信度: +{b1_drop:.1%} | 外枠穴: {outer_boat}号艇 (+{outer_ov:.1%}) | 投資: {total_bet:,}円")
        print(bets_formatted_table)
        print("=" * 65 + "\n")
        return True

    try:
        resp = requests.post(webhook_url, json={"embeds": [embed]}, timeout=10)
        if resp.status_code in (200, 204):
            logger.info("✅ 万舟スナイパー Discord Webhook 送信成功！")
            return True
        else:
            logger.warning(f"⚠️ 万舟スナイパー Discord Webhook 送信失敗 (Status: {resp.status_code}): {resp.text}")
            return False
    except Exception as e:
        logger.error(f"❌ 万舟スナイパー Discord Webhook 送信中にエラーが発生しました: {e}")
        return False


def calculate_manshu_sniper_bets(
    p1_dict_residual: Dict[int, float],
    p_norm_syn_dict: Dict[int, float],
    all_odds: Dict[str, float],
    d2: float = 0.60,
    d3: float = 0.80,
    min_odds: float = 100.0,
    max_odds: float = 500.0,
    min_ev: float = 1.20,
    max_combos: int = 8
) -> Dict[str, Any]:
    """
    万舟スナイパー (💣 万舟スナイパー) の発動判定と100円均等買い目(最大8点)を算出。
    - Tier 1 (🔥 激熱万舟): b1_drop >= 0.06 かつ outer_ov >= 0.05 (2026年OOT: 万舟的中率10.5%, 回収率443.5%)
    - Tier 2 (💣 万舟好機): b1_drop >= 0.05 かつ outer_ov >= 0.04 (2026年OOT: 万舟的中率5.9%, 回収率243.2%)
    """
    b1_drop = float(p_norm_syn_dict.get(1, 0.0)) - float(p1_dict_residual.get(1, 0.0))
    outer_diffs = {
        b: float(p1_dict_residual.get(b, 0.0)) - float(p_norm_syn_dict.get(b, 0.0))
        for b in (4, 5, 6)
    }
    outer_boat = max(outer_diffs, key=outer_diffs.get) if outer_diffs else 4
    outer_ov = float(outer_diffs.get(outer_boat, 0.0))

    manshu_benter_list, _, _ = calculate_benter_probs(
        p1_dict_residual,
        d2=d2,
        d3=d3,
        calibration_method='direct'
    )
    manshu_probs_dict = {p['combo']: p['prob'] for p in manshu_benter_list}

    candidate_combos = []
    for combo, prob in manshu_probs_dict.items():
        o_val = float(all_odds.get(combo, 0.0))
        if min_odds <= o_val <= max_odds and prob > 0:
            ev = prob * o_val
            if ev >= min_ev:
                candidate_combos.append((combo, prob, o_val, ev))

    candidate_combos.sort(key=lambda x: x[1], reverse=True)
    selected = candidate_combos[:max_combos]

    is_tier1_cond = (b1_drop >= 0.06) and (outer_ov >= 0.05) and (len(selected) > 0)
    is_tier2_cond = (not is_tier1_cond) and (b1_drop >= 0.05) and (outer_ov >= 0.04) and (len(selected) > 0)
    is_manshu_go = is_tier1_cond or is_tier2_cond

    bets_manshu = {c: 100 for c, _, _, _ in selected} if is_manshu_go else {}
    cum_prob = sum(p for _, p, _, _ in selected) if is_manshu_go else 0.0
    exp_roi = (sum(p * o * 100.0 for _, p, o, _ in selected) / (len(selected) * 100.0)) if (is_manshu_go and selected) else 0.0
    inv_sum = sum(1.0 / o for _, _, o, _ in selected if o > 0) if is_manshu_go else 0.0
    syn_odds = (1.0 / inv_sum) if inv_sum > 0 else 0.0

    return {
        'is_manshu_go': is_manshu_go,
        'is_tier1': is_tier1_cond,
        'is_tier2': is_tier2_cond,
        'b1_drop': b1_drop,
        'outer_ov': outer_ov,
        'outer_boat': outer_boat,
        'bets': bets_manshu,
        'probs_dict': manshu_probs_dict,
        'cum_prob': cum_prob,
        'expected_roi': exp_roi,
        'synthetic_odds': syn_odds
    }


# =====================================================================
# 4. 黄金ベースライン レース評価エンジン
# =====================================================================

def calculate_dutching_bets(
    benter_probs: Dict[str, float],
    odds_dict: Dict[str, float],
    budget: int = 1000,
    target_cum_prob: float = 0.50,
    max_combos: int = 8,
    min_combos: int = 2,
    min_synthetic_odds: Optional[float] = None
) -> Dict[str, int]:
    """
    的中特化: 累積確率50%（最大8点）を勝率順に抽出し、
    合成オッズ制約（min_synthetic_odds）を排して完全な的中率特化で買い目を選定。
    オッズ逆数比による動的ダッチング資金配分とトリガミ回避ループを実行。
    """
    valid_combos = []
    for combo, prob in sorted(benter_probs.items(), key=lambda x: x[1], reverse=True):
        o = odds_dict.get(combo, 0.0)
        if o > 1.0 and prob > 0:
            valid_combos.append((combo, prob, o))
            
    if not valid_combos:
        return {}
        
    selected = []
    cum_p = 0.0
    inv_sum = 0.0
    max_inv_sum = (1.0 / min_synthetic_odds) if (min_synthetic_odds is not None and min_synthetic_odds > 0) else 0.99
    
    for item in valid_combos:
        c_name, c_prob, c_odds = item
        next_inv_sum = inv_sum + (1.0 / c_odds)
        
        # 合成オッズ制約がある場合のみチェック（未指定時は0.99まで許容して的中率を最大化）
        if selected and next_inv_sum > max_inv_sum:
            if len(selected) >= min_combos:
                break
            elif len(selected) + 1 >= min_combos and next_inv_sum <= 0.90:
                selected.append(item)
                cum_p += c_prob
                inv_sum = next_inv_sum
                break
            else:
                break
                
        selected.append(item)
        cum_p += c_prob
        inv_sum = next_inv_sum
        
        if (cum_p >= target_cum_prob or len(selected) >= max_combos) and len(selected) >= min_combos:
            break
            
    if len(selected) < min_combos:
        selected = valid_combos[:min(len(valid_combos), min_combos)]
        
    while len(selected) > min_combos:
        s_val = sum(1.0 / o for _, _, o in selected)
        if s_val <= max_inv_sum:
            break
        selected.pop()
        
    def allocate(combos_list, target_budget):
        s = sum(1.0 / o for _, _, o in combos_list)
        if s <= 0:
            return {c: 100 for c, _, _ in combos_list}
        res = {}
        for c, _, o in combos_list:
            raw = target_budget * ((1.0 / o) / s)
            amt = max(100, int(round(raw / 100.0)) * 100)
            res[c] = amt
        return res
        
    bets = allocate(selected, budget)
    
    for _ in range(5):
        tot = sum(bets.values())
        trigami_found = False
        for c, _, o in list(selected):
            payout = bets.get(c, 0) * o
            if payout <= tot:
                trigami_found = True
                if (bets[c] + 100) * o > (tot + 100):
                    bets[c] += 100
                elif len(selected) > min_combos:
                    selected.pop()
                    bets = allocate(selected, budget)
                    break
        if not trigami_found:
            break
            
    return bets


def export_radar_state(
    race_info: Dict[str, Any],
    ai_eval: Dict[str, Any],
    df_feat: Optional[pd.DataFrame] = None,
    df_race: Optional[pd.DataFrame] = None,
    p1_dict_honmei: Optional[Dict[int, float]] = None,
    all_odds: Optional[Dict[str, float]] = None,
    benter_probs_dict: Optional[Dict[str, float]] = None,
    bets: Optional[Dict[str, int]] = None,
    bets_hit_focused: Optional[Dict[str, int]] = None
) -> None:
    """最新の推論・全評価状態を階層構造の current_radar.json に上書き出力"""
    try:
        radar_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'current_radar.json')

        # 1. 出走表 & 直前情報（1〜6号艇）の構築
        boats_list = []
        for bn in range(1, 7):
            b_dict = {
                'boat_number': bn,
                'racer_id': 0,
                'racer_name': f"{bn}号艇",
                'motor_rate': 0.0,
                'boat_rate': 0.0,
                'ex_time': 0.0,
                'tilt': -0.5,
                'st_time': 0.20,
                'pred_course': bn,
                'ex_momentum_diff': 0.0,
                'p1_prob': float(p1_dict_honmei.get(bn, 0.0)) if p1_dict_honmei else 0.0
            }

            source_df = df_feat if (df_feat is not None and not df_feat.empty) else df_race
            if source_df is not None and not source_df.empty:
                b_match = source_df[source_df['boat_number'] == bn]
                if len(b_match) > 0:
                    row = b_match.iloc[0]
                    if 'racer_id' in row and pd.notna(row['racer_id']):
                        b_dict['racer_id'] = int(row['racer_id'])
                    if 'racer_name' in row and pd.notna(row['racer_name']):
                        b_dict['racer_name'] = str(row['racer_name'])
                    if 'motor_rate' in row and pd.notna(row['motor_rate']):
                        b_dict['motor_rate'] = float(row['motor_rate'])
                    if 'boat_rate' in row and pd.notna(row['boat_rate']):
                        b_dict['boat_rate'] = float(row['boat_rate'])
                    if 'exhibition_time' in row and pd.notna(row['exhibition_time']):
                        b_dict['ex_time'] = float(row['exhibition_time'])
                    if 'tilt' in row and pd.notna(row['tilt']):
                        b_dict['tilt'] = float(row['tilt'])
                    if 'exhibition_start_timing' in row and pd.notna(row['exhibition_start_timing']):
                        b_dict['st_time'] = float(row['exhibition_start_timing'])
                    if 'pred_course' in row and pd.notna(row['pred_course']):
                        b_dict['pred_course'] = int(row['pred_course'])
                    if 'ex_momentum_diff' in row and pd.notna(row['ex_momentum_diff']):
                        b_dict['ex_momentum_diff'] = float(row['ex_momentum_diff'])

            boats_list.append(b_dict)

        # 2. 黄金ベースライン (Sniper) オッズリストの構築
        odds_golden = []
        if all_odds:
            candidate_combos = []
            if benter_probs_dict:
                for combo, prob in benter_probs_dict.items():
                    o_val = all_odds.get(combo, 0.0)
                    ev_val = prob * o_val
                    rec_amt = bets.get(combo, 0) if bets else 0
                    candidate_combos.append({
                        'combo': combo,
                        'odds': o_val,
                        'prob': prob,
                        'ev': ev_val,
                        'recommended_amount': rec_amt,
                        'expected_return': int((rec_amt / 100.0) * o_val * 100.0) if rec_amt > 0 else 0,
                        'is_recommended': rec_amt > 0
                    })
                candidate_combos.sort(key=lambda x: (x['is_recommended'], x['ev']), reverse=True)
                odds_golden = candidate_combos[:15]
            else:
                for combo, o_val in sorted(all_odds.items(), key=lambda x: x[1])[:10]:
                    odds_golden.append({
                        'combo': combo,
                        'odds': o_val,
                        'prob': 0.0,
                        'ev': 0.0,
                        'recommended_amount': 0,
                        'expected_return': 0,
                        'is_recommended': False
                    })

        # 3. 的中特化 (Dutching) オッズリストの構築
        odds_hit_focused = []
        if all_odds and benter_probs_dict:
            total_hit_bet = sum(bets_hit_focused.values()) if bets_hit_focused else 0
            candidate_hit = []
            for combo, prob in benter_probs_dict.items():
                o_val = all_odds.get(combo, 0.0)
                ev_val = prob * o_val
                rec_amt = bets_hit_focused.get(combo, 0) if bets_hit_focused else 0
                exp_ret = int((rec_amt / 100.0) * o_val * 100.0) if rec_amt > 0 else 0
                profit = (exp_ret - total_hit_bet) if rec_amt > 0 else 0
                candidate_hit.append({
                    'combo': combo,
                    'odds': o_val,
                    'prob': prob,
                    'ev': ev_val,
                    'recommended_amount': rec_amt,
                    'expected_return': exp_ret,
                    'profit': profit,
                    'is_recommended': rec_amt > 0
                })
            candidate_hit.sort(key=lambda x: (x['is_recommended'], x['prob']), reverse=True)
            odds_hit_focused = candidate_hit[:15]

        # 4. 全体オッズ（人気上位30件）の構築
        all_odds_list = []
        if all_odds:
            for combo, o_val in sorted(all_odds.items(), key=lambda x: x[1])[:30]:
                p_val = benter_probs_dict.get(combo, 0.0) if benter_probs_dict else 0.0
                ev_val = p_val * o_val
                all_odds_list.append({
                    'combo': combo,
                    'odds': o_val,
                    'prob': p_val,
                    'ev': ev_val
                })

        total_bet_golden = sum(bets.values()) if bets else 0
        total_bet_hit = sum(bets_hit_focused.values()) if bets_hit_focused else 0

        payload = {
            'race': race_info,
            'ai_eval': {
                **ai_eval,
                'total_bet_golden': total_bet_golden,
                'bets_count_golden': len(bets) if bets else 0,
                'total_bet_hit': total_bet_hit,
                'bets_count_hit': len(bets_hit_focused) if bets_hit_focused else 0,
            },
            'boats': boats_list,
            'all_odds': all_odds_list,
            'odds': odds_golden,
            'odds_golden': odds_golden,
            'odds_hit_focused': odds_hit_focused,
            # フラット互換
            'race_name': race_info.get('race_name', ''),
            'venue_name': race_info.get('venue_name', ''),
            'race_no': race_info.get('race_no', 0),
            'deadline': race_info.get('deadline', ''),
            'p1_score': ai_eval.get('p1_score', 0.0),
            'top_boat': ai_eval.get('top_boat', 1),
            'gatekeeper_passed': ai_eval.get('gatekeeper_passed', False),
            'motor_rate': ai_eval.get('motor_rate', 0.0),
            'wave_height': ai_eval.get('wave_height', 0.0),
            'sniper_go': ai_eval.get('sniper_go', False),
            'sniper_node': ai_eval.get('sniper_node', ''),
            'status': ai_eval.get('status', ''),
            'status_message': ai_eval.get('status_message', ''),
            'total_bet': total_bet_golden,
            'bets_count': len(bets) if bets else 0,
            'bets': [
                {
                    'combo': o['combo'],
                    'amount': o['recommended_amount'],
                    'odds': o['odds'],
                    'ev': o['ev'],
                    'expected_return': o['expected_return']
                }
                for o in odds_golden if o.get('is_recommended')
            ],
            'updated_at': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 1. 単体 current_radar.json に保存
        with open(radar_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 2. 複数レースストック recent_radars.json に保存（最大5件）
        recent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recent_radars.json')
        recent_list = []
        if os.path.exists(recent_path):
            try:
                with open(recent_path, 'r', encoding='utf-8') as rf:
                    loaded = json.load(rf)
                    if isinstance(loaded, list):
                        recent_list = loaded
            except Exception:
                recent_list = []

        # 重複レースの除外（同一会場・レース番号の古い推論を更新）
        current_race_key = f"{race_info.get('venue_name')}_{race_info.get('race_no')}"
        recent_list = [
            r for r in recent_list 
            if f"{r.get('race', {}).get('venue_name', r.get('venue_name', ''))}_{r.get('race', {}).get('race_no', r.get('race_no', 0))}" != current_race_key
        ]
        recent_list.insert(0, payload)
        recent_list = recent_list[:5]

        with open(recent_path, 'w', encoding='utf-8') as rf:
            json.dump(recent_list, rf, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.debug(f"radar_state 出力例外: {e}")


def evaluate_race(
    date_str: str,
    venue_code: int,
    venue_name: str,
    race_no: int,
    deadline_str: str,
    bankroll: float = DEFAULT_BANKROLL,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_exposure: float = DEFAULT_MAX_EXPOSURE,
    max_concentration: float = DEFAULT_MAX_CONCENTRATION,
    min_ev: float = DEFAULT_MIN_EV,
    max_odds: float = DEFAULT_MAX_ODDS,
    gatekeeper_th: float = DEFAULT_GATEKEEPER_TH,
    webhook_url: str = DISCORD_WEBHOOK_URL,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    指定レースの直前オッズ・出走データを取得し、黄金ベースライン推論を実行
    """
    race_id = f"{date_str}_{venue_code}_{race_no}"
    race_name = f"{venue_name} {race_no}R"
    logger.info(f"🔍 [EVALUATING] {race_name} (締切: {deadline_str}) の分析を開始...")
    
    cluster_d2, cluster_d3, cluster_id, cluster_name = get_cluster_benter_params(venue_code)
    race_info = {
        'race_name': race_name,
        'venue_name': venue_name,
        'venue_code': venue_code,
        'race_no': race_no,
        'deadline': deadline_str,
        'cluster_id': cluster_id,
        'cluster_name': cluster_name
    }

    # 1. 会場クラスタ判定 (Regime A/B の1号艇アタマは Cluster 1 を除外、Regime C の逆転スナイプおよび的中特化は全24会場対応)
    is_cluster1_venue = (venue_code in CLUSTER_1_VENUES)
        
    # 2. 出走データ & 直前オッズスクレイピング
    df_race = BoatRaceScraper.get_race_data(date_str, venue_code, race_no)
    all_odds = BoatRaceScraper.get_odds(date_str, venue_code, race_no)
    
    if df_race is None or df_race.empty:
        logger.warning(f"⏳ [{venue_name} {race_no}R] 展示データ・出走表が未発表または取得できませんでした（通常、締切約20分前に公開されます）。")
        return {'status': 'error_data'}
        
    if not all_odds:
        logger.warning(f"⏳ [{venue_name} {race_no}R] 直前3連単オッズが未発表または取得できませんでした。")
        return {'status': 'error_odds'}
        
    # オッズデータを Supabase に非同期/即時保存
    try:
        db_manager.save_odds_batch(race_id, all_odds)
    except Exception as e:
        logger.debug(f"オッズ保存エラー: {e}")
        
    # 3. 特徴量エンジニアリング & 欠場艇動的補正
    df_feat = FeatureEngineer.process(df_race, venue_name, race_date=date_str)
    active_boats = df_feat['boat_number'].tolist()

    absent_boats = sorted(list(set(range(1, 7)) - set(active_boats)))
    if absent_boats:
        absent_str = "、".join([f"{b}号艇" for b in absent_boats])
        logger.info(f"📢 [{venue_name} {race_no}R] 欠場検知: {absent_str} 除外 ({len(active_boats)}艇立てで処理)")
        
    # 4. オッズ合成勝率 & init_score ロジット変換
    syn_dict = {b: 0.0 for b in active_boats}
    for combo, o_val in all_odds.items():
        try:
            parts = combo.split('-')
            b1 = int(parts[0])
            if o_val > 0 and b1 in syn_dict:
                syn_dict[b1] += (1.0 / o_val)
        except Exception:
            pass
            
    total_syn = sum(syn_dict.values())
    if total_syn > 0:
        p_norm_syn_dict = {b: syn_dict[b] / total_syn for b in active_boats}
    else:
        p_norm_syn_dict = {b: 1.0 / len(active_boats) for b in active_boats}
        
    # 学習時 (train_data_full.csv) と同一の未正規化オッズ逆数和を syn_win_rate に設定し、init_score には正規化確率を使用
    df_feat['syn_win_rate'] = df_feat['boat_number'].map(syn_dict).fillna(0.0)
    p_norm_syn_series = df_feat['boat_number'].map(p_norm_syn_dict).fillna(1.0 / len(active_boats))
    df_feat['init_score'] = probs_to_init_scores(p_norm_syn_series.to_numpy())
    
    # 5. Gatekeeper 推論 (Honmei + Platt Scaling)
    if not os.path.exists(MODEL_HONMEI_PATH) or not os.path.exists(MODEL_RESIDUAL_PATH):
        logger.error(f"❌ モデルファイルが見つかりません ({MODEL_HONMEI_PATH}, {MODEL_RESIDUAL_PATH})")
        return {'status': 'error_model'}
        
    model_h = lgb.Booster(model_file=MODEL_HONMEI_PATH)
    df_h = prepare_features_for_model(df_feat, model_h)
    df_feat['score_honmei'] = model_h.predict(df_h)
    
    calibrator = get_default_calibrator('platt')
    scores_h_dict = dict(zip(df_feat['boat_number'], df_feat['score_honmei']))
    p1_dict_honmei = calibrator.calibrate_scores(scores_h_dict)
    
    sorted_p1 = sorted(p1_dict_honmei.items(), key=lambda x: x[1], reverse=True)
    top_boat, max_p1 = sorted_p1[0]
    prob_gap = (max_p1 - sorted_p1[1][1]) if len(sorted_p1) > 1 else 0.0
    
    logger.info(f"🛡️ [{venue_name} {race_no}R] Gatekeeper P1 = {max_p1:.2%} ({top_boat}号艇本命 / 閾値: {gatekeeper_th:.2%})")
    
    b1_feat = df_feat[df_feat['boat_number'] == 1]
    b1_motor_rate = float(b1_feat['motor_rate'].iloc[0]) if (len(b1_feat) > 0 and 'motor_rate' in b1_feat.columns and pd.notna(b1_feat['motor_rate'].iloc[0])) else 0.0
    wave_height = float(df_feat['wave_height'].iloc[0]) if ('wave_height' in df_feat.columns and pd.notna(df_feat['wave_height'].iloc[0])) else 0.0

    # 6. Extractor 推論 (Residual 1着 & 条件付き2・3着モデル)
    model_r = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
    df_r = prepare_features_for_model(df_feat, model_r)
    raw_res = model_r.predict(df_r, raw_score=True)
    total_logits = raw_res + df_feat['init_score'].to_numpy()
    p_raw_res = 1.0 / (1.0 + np.exp(-np.clip(total_logits, -30, 30)))
    p_norm_res = p_raw_res / np.sum(p_raw_res)
    p1_dict_residual = dict(zip(df_feat['boat_number'], p_norm_res))

    b1_p_res = float(p1_dict_residual.get(1, 0.0))
    b1_p_mkt = float(p_norm_syn_dict.get(1, 0.0))

    # 5.5 新戦略 Triple-Regime Gatekeeper & Sniper Filter (2026年OOT全4四半期黒字・ROI 159.79% / +469,300円)
    is_regime_a = (not is_cluster1_venue) and (top_boat == 1 and max_p1 >= 0.80 and b1_p_res > b1_p_mkt)
    is_regime_b = (not is_cluster1_venue) and (top_boat == 1 and b1_p_res >= 0.74 and b1_motor_rate >= 35.0)
    regime_c_heads = tuple(
        b for b in (2, 3, 4)
        if float(p1_dict_residual.get(b, 0.0)) >= 0.35
        and (float(p1_dict_residual.get(b, 0.0)) - float(p_norm_syn_dict.get(b, 0.0))) >= 0.11
    )
    is_regime_c = (len(regime_c_heads) > 0) and not (is_regime_a or is_regime_b)
    gatekeeper_passed = ((not is_cluster1_venue) and (top_boat == 1) and ((max_p1 >= gatekeeper_th) or (b1_p_res >= 0.74))) or is_regime_c

    sniper_go = False
    sniper_node = ""
    hybrid_alpha = 0.0
    eff_d2, eff_d3 = (0.70, 0.80) if is_regime_c else (cluster_d2, cluster_d3)
    if is_regime_a:
        sniper_go = True
        sniper_node = "Regime A: 1号艇・市場過小評価オーバーレイ (Hybrid 2nd/3rd)"
        hybrid_alpha = 0.6
    elif is_regime_b:
        sniper_go = True
        sniper_node = "Regime B: 1号艇・高信頼残差×好モーター (Benter)"
        hybrid_alpha = 0.0
    elif is_regime_c:
        sniper_go = True
        heads_str = ",".join(f"{b}号艇" for b in regime_c_heads)
        sniper_node = f"Regime C: 逆転スナイプ [{heads_str}→1号艇2着残し] (Overlay>=11%)"
        hybrid_alpha = 0.0

    # 7. 会場クラスタ別 Benter 確率展開 & 条件付き2・3着ハイブリッド統合
    benter_probs, _, _ = calculate_benter_probs(
        p1_dict_residual,
        d2=eff_d2,
        d3=eff_d3,
        calibration_method='direct'
    )
    benter_probs_dict = {p['combo']: p['prob'] for p in benter_probs}

    if hybrid_alpha > 0.0 and os.path.exists(MODEL_2ND_PATH) and os.path.exists(MODEL_3RD_PATH):
        try:
            model_2nd = lgb.Booster(model_file=MODEL_2ND_PATH)
            model_3rd = lgb.Booster(model_file=MODEL_3RD_PATH)
            sub_sum = float(df_feat.loc[df_feat['boat_number'] >= 2, 'syn_win_rate'].sum())
            p_sub = np.where(sub_sum > 0, df_feat['syn_win_rate'].to_numpy() / max(sub_sum, 1e-9), 0.20)
            init_sub = probs_to_init_scores(p_sub, clip_eps=1e-4)
            init_23 = probs_to_init_scores(np.clip(p_sub * 2.0, 1e-4, 1.0 - 1e-4), clip_eps=1e-4)
            df_2 = prepare_features_for_model(df_feat, model_2nd)
            p_c2 = 1.0 / (1.0 + np.exp(-np.clip(model_2nd.predict(df_2, raw_score=True) + init_sub, -30, 30)))
            p_c3 = 1.0 / (1.0 + np.exp(-np.clip(model_3rd.predict(df_2, raw_score=True) + init_23, -30, 30)))
            boats_arr = df_feat['boat_number'].to_numpy(dtype=int)
            cond2_d = {int(boats_arr[i]): max(float(p_c2[i]), 1e-9) for i in range(len(boats_arr))}
            cond3_d = {int(boats_arr[i]): max(float(p_c3[i]), 1e-9) for i in range(len(boats_arr))}
            for b1, b2, b3 in itertools.permutations(boats_arr, 3):
                c_key = f"{int(b1)}-{int(b2)}-{int(b3)}"
                prob2_c = cond2_d[int(b2)] / sum(cond2_d[int(b)] for b in boats_arr if b != b1)
                prob3_c = cond3_d[int(b3)] / sum(cond3_d[int(b)] for b in boats_arr if b != b1 and b != b2)
                p_cond = float(p1_dict_residual.get(int(b1), 0.0)) * prob2_c * prob3_c
                benter_probs_dict[c_key] = (1.0 - hybrid_alpha) * benter_probs_dict.get(c_key, 0.0) + hybrid_alpha * p_cond
        except Exception as e:
            logger.warning(f"条件付き2・3着モデル統合フォールバック: {e}")
    
    # 8. 的中特化（動的ダッチング）をメモリ上で計算 (新モデル＆条件付き2・3着モデル最適化版)
    bets_hit_focused = {}
    exp_roi_hit = 0.0
    syn_odds_hit = 0.0
    is_gogo_lamp = False
    is_chance_lamp = False
    dutching_probs_dict = dict(benter_probs_dict)
    if (not is_cluster1_venue) and p1_dict_residual and all_odds:
        try:
            # 的中特化ダッチング用の確率展開: GOGO候補は (C0:0.50/0.75, C2:0.10/0.25)、CHANCE候補は条件付き2・3着ハイブリッド (d=0.50/0.75, alpha=0.6)
            is_gogo_cand = (top_boat == 1) and (max_p1 is not None and max_p1 >= 0.65) and (b1_p_res > b1_p_mkt) and (b1_motor_rate >= 40.0)
            d_hit2, d_hit3 = ((0.50, 0.75) if cluster_id == 0 else (0.10, 0.25)) if is_gogo_cand else (0.50, 0.75)
            hit_benter_list, _, _ = calculate_benter_probs(p1_dict_residual, d2=d_hit2, d3=d_hit3, calibration_method='direct')
            dutching_probs_dict = {p['combo']: p['prob'] for p in hit_benter_list}

            if (not is_gogo_cand) and os.path.exists(MODEL_2ND_PATH) and os.path.exists(MODEL_3RD_PATH):
                try:
                    m2_h = lgb.Booster(model_file=MODEL_2ND_PATH)
                    m3_h = lgb.Booster(model_file=MODEL_3RD_PATH)
                    sub_s = float(df_feat.loc[df_feat['boat_number'] >= 2, 'syn_win_rate'].sum())
                    p_s = np.where(sub_s > 0, df_feat['syn_win_rate'].to_numpy() / max(sub_s, 1e-9), 0.20)
                    in_s2 = probs_to_init_scores(p_s, clip_eps=1e-4)
                    in_s3 = probs_to_init_scores(np.clip(p_s * 2.0, 1e-4, 1.0 - 1e-4), clip_eps=1e-4)
                    df_2h = prepare_features_for_model(df_feat, m2_h)
                    pc2_h = 1.0 / (1.0 + np.exp(-np.clip(m2_h.predict(df_2h, raw_score=True) + in_s2, -30, 30)))
                    pc3_h = 1.0 / (1.0 + np.exp(-np.clip(m3_h.predict(df_2h, raw_score=True) + in_s3, -30, 30)))
                    b_arr = df_feat['boat_number'].to_numpy(dtype=int)
                    c2_map = {int(b_arr[i]): max(float(pc2_h[i]), 1e-9) for i in range(len(b_arr))}
                    c3_map = {int(b_arr[i]): max(float(pc3_h[i]), 1e-9) for i in range(len(b_arr))}
                    for b1, b2, b3 in itertools.permutations(b_arr, 3):
                        if int(b1) == 1:
                            ck = f"{int(b1)}-{int(b2)}-{int(b3)}"
                            pr2 = c2_map[int(b2)] / sum(c2_map[int(b)] for b in b_arr if b != b1)
                            pr3 = c3_map[int(b3)] / sum(c3_map[int(b)] for b in b_arr if b != b1 and b != b2)
                            p_c = float(p1_dict_residual.get(1, 0.0)) * pr2 * pr3
                            dutching_probs_dict[ck] = 0.4 * dutching_probs_dict.get(ck, 0.0) + 0.6 * p_c
                except Exception:
                    pass

            bets_hit_focused = calculate_dutching_bets(
                benter_probs=dutching_probs_dict,
                odds_dict=all_odds,
                budget=1000,
                target_cum_prob=0.50,
                max_combos=8,
                min_combos=2,
                min_synthetic_odds=None
            )
            if bets_hit_focused:
                tot_hit_b = sum(bets_hit_focused.values())
                exp_pay = sum(
                    dutching_probs_dict.get(c, 0.0) * all_odds.get(c, 0.0) * amt
                    for c, amt in bets_hit_focused.items()
                )
                exp_roi_hit = (exp_pay / tot_hit_b) if tot_hit_b > 0 else 0.0
                inv_sum = sum(1.0 / all_odds.get(c, 1.0) for c in bets_hit_focused.keys() if all_odds.get(c, 0) > 0)
                syn_odds_hit = 1.0 / inv_sum if inv_sum > 0 else 0.0

                # 🎰 GOGO!ランプ点灯判定 (新モデル対応): 1号艇本命 & P1>=65% & 残差オーバーレイ(Res>Mkt) & モーター>=40% & 期待回収率>=1.10
                # 2026年OOT実績: 170R / 91的中 (的中率53.5%) / 回収率104.2% (+8,701円)
                if is_gogo_cand and (exp_roi_hit >= 1.10):
                    is_gogo_lamp = True
                # ⭐ CHANCEランプ点灯判定 (新モデル＋2・3着モデル対応): 1号艇本命 & P1>=60% & モーター>=36% & 合成オッズ>=2.0 & 期待回収率>=1.05
                # 2026年OOT実績: 198R / 86的中 (的中率43.4%) / 回収率105.5% (+11,568円)
                elif (top_boat == 1) and (max_p1 is not None and max_p1 >= 0.60) and (b1_motor_rate >= 36.0) and (syn_odds_hit >= 2.0) and (exp_roi_hit >= 1.05):
                    is_chance_lamp = True
        except Exception: pass

    # 9. ポートフォリオ最適化 (Regime A/B: 1-2/3/4-x, Regime C: 2/3/4-1-x スジ舟券フィルタ適用)
    if is_regime_c:
        allowed_h_strs = {str(b) for b in regime_c_heads}
        sniper_probs_dict = {
            c: (p if len(c.split('-')) == 3 and c.split('-')[0] in allowed_h_strs and c.split('-')[1] == '1' else 0.0)
            for c, p in benter_probs_dict.items()
        }
    else:
        sniper_probs_dict = {
            c: (p if len(c.split('-')) == 3 and c.split('-')[0] == '1' and c.split('-')[1] in ('2', '3', '4') else 0.0)
            for c, p in benter_probs_dict.items()
        }
    optimizer = PortfolioOptimizer()
    bets = optimizer.optimize_funds(
        probabilities=sniper_probs_dict,
        odds=all_odds,
        bankroll=float(bankroll),
        risk_aversion=float(risk_aversion),
        max_exposure=max_exposure,
        max_concentration=max_concentration,
        min_ev=float(min_ev),
        max_odds=float(max_odds),
        kelly_fraction=None  # 固定ウェイト (レース上限5%, 買い目上限2%)
    )

    # 10. 判定ステータスと投資フラグの決定
    if not gatekeeper_passed:
        status = "gatekeeper_skipped"
        status_message = f"☕ Gatekeeper未達 (P1={max_p1:.1%} < {gatekeeper_th:.1%})"
        is_investment_go = False
    elif not sniper_go:
        status = "sniper_skipped"
        status_message = f"🎯 Sniper Filter見送り (P1={max_p1:.1%}, Motor={b1_motor_rate:.1f}%, Wave={wave_height:.1f}cm)"
        is_investment_go = False
    elif not bets:
        status = "no_value_bets"
        status_message = f"🔍 EV条件未達見送り [{sniper_node}] (EV >= {min_ev:.2f} 買い目なし)"
        is_investment_go = False
    else:
        status = "investment_go"
        status_message = f"🚀 投資GOサイン点灯！ [{sniper_node}] ({len(bets)}点 / {sum(bets.values()):,}円)"
        is_investment_go = True

    # 11. ライブモニター用 JSON エクスポート (全レースで計算結果を出力)
    export_radar_state(
        race_info=race_info,
        ai_eval={
            'p1_score': max_p1,
            'top_boat': top_boat,
            'prob_gap': prob_gap,
            'gatekeeper_passed': gatekeeper_passed,
            'motor_rate': b1_motor_rate,
            'wave_height': wave_height,
            'sniper_go': sniper_go,
            'sniper_node': sniper_node,
            'status': status,
            'status_message': status_message,
            'total_bet': sum(bets.values()) if is_investment_go else 0,
            'bets_count': len(bets) if is_investment_go else 0,
            'is_gogo_lamp': is_gogo_lamp,
            'is_chance_lamp': is_chance_lamp,
            'expected_roi': exp_roi_hit,
            'synthetic_odds': syn_odds_hit
        },
        df_feat=df_feat,
        df_race=df_race,
        p1_dict_honmei=p1_dict_honmei,
        all_odds=all_odds,
        benter_probs_dict=benter_probs_dict,
        bets=bets if is_investment_go else {},
        bets_hit_focused=bets_hit_focused
    )

    # 11.5 【全レース蓄積】的中特化（ダッチング）データを専用DBテーブルへ常時保存
    if bets_hit_focused:
        try:
            db_manager.save_hit_focused_prediction(
                race_id=race_id,
                race_date=date_str,
                venue_code=venue_code,
                venue_name=venue_name,
                race_no=race_no,
                deadline_time=deadline_str,
                top_boat=top_boat,
                max_p1=max_p1,
                prob_gap=prob_gap,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                status=status,
                bets=bets_hit_focused,
                benter_probs=benter_probs_dict,
                all_odds=all_odds,
                target_cum_prob=0.50,
                is_gogo_lamp=is_gogo_lamp,
                is_chance_lamp=is_chance_lamp,
                expected_roi=exp_roi_hit,
                motor_rate=b1_motor_rate,
                synthetic_odds=syn_odds_hit
            )
            lamp_badge = ""
            if is_gogo_lamp:
                lamp_badge = " 🔥【GOGO!ランプ点灯】(回収率110.6%)"
            elif is_chance_lamp:
                lamp_badge = " ⭐【CHANCEランプ点灯】(回収率101.3%)"
            logger.info(f"💾 [{venue_name} {race_no}R] 的中特化データを hit_focused_predictions に自動蓄積保存しました ({len(bets_hit_focused)}点 / {sum(bets_hit_focused.values()):,}円 / ExpROI={exp_roi_hit:.2f} / 合成={syn_odds_hit:.2f}){lamp_badge}")
        except Exception as e:
            logger.warning(f"的中特化データの保存エラー: {e}")

    # 11.8 【万舟スナイパー (💣 万舟スナイパー)】算出・DB保存・Discord通知
    try:
        manshu_res = calculate_manshu_sniper_bets(
            p1_dict_residual=p1_dict_residual,
            p_norm_syn_dict=p_norm_syn_dict,
            all_odds=all_odds,
            d2=0.60,
            d3=0.80,
            min_odds=100.0,
            max_odds=500.0,
            min_ev=1.20,
            max_combos=8
        )
        is_manshu_go = manshu_res['is_manshu_go']
        manshu_status = "manshu_go" if is_manshu_go else "manshu_skipped"
        db_manager.save_manshu_prediction(
            race_id=race_id,
            race_date=date_str,
            venue_code=venue_code,
            venue_name=venue_name,
            race_no=race_no,
            deadline_time=deadline_str,
            top_boat=manshu_res['outer_boat'] if is_manshu_go else top_boat,
            max_p1=max_p1,
            prob_gap=prob_gap,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            status=manshu_status,
            bets=manshu_res['bets'],
            benter_probs=manshu_res['probs_dict'],
            all_odds=all_odds,
            target_cum_prob=manshu_res['cum_prob'],
            is_gogo_lamp=manshu_res['is_tier1'],
            is_chance_lamp=manshu_res['is_tier2'],
            expected_roi=manshu_res['expected_roi'],
            motor_rate=b1_motor_rate,
            synthetic_odds=manshu_res['synthetic_odds'],
            b1_drop=manshu_res['b1_drop'],
            outer_ov=manshu_res['outer_ov']
        )
        if is_manshu_go:
            tier_label = "🔥【激熱万舟 Tier1 (回収率443.5%)】" if manshu_res['is_tier1'] else "💣【万舟好機 Tier2 (回収率243.2%)】"
            logger.info(
                f"💣 [{venue_name} {race_no}R] 万舟スナイパー発動！ {tier_label} "
                f"(1号艇過信:{manshu_res['b1_drop']:+.1%}, {manshu_res['outer_boat']}号艇穴:{manshu_res['outer_ov']:+.1%}, "
                f"{len(manshu_res['bets'])}点 / {sum(manshu_res['bets'].values()):,}円)"
            )
            send_manshu_discord_notification(
                webhook_url=webhook_url,
                date_str=date_str,
                venue_code=venue_code,
                venue_name=venue_name,
                race_no=race_no,
                deadline_str=deadline_str,
                is_tier1=manshu_res['is_tier1'],
                b1_drop=manshu_res['b1_drop'],
                outer_ov=manshu_res['outer_ov'],
                outer_boat=manshu_res['outer_boat'],
                bets=manshu_res['bets'],
                manshu_probs=manshu_res['probs_dict'],
                all_odds=all_odds,
                dry_run=dry_run
            )
    except Exception as e:
        logger.warning(f"万舟スナイパーデータの保存・通知エラー: {e}")

    # 12. Supabase DB保存 & Discord通知 (黄金ベースライン)
    # 本番用の race_predictions テーブルへ全評価レースの推論結果・判定ステータスを保存（INSERT/UPSERT）
    try:
        db_manager.save_race_prediction(
            race_id=race_id, race_date=date_str, venue_code=venue_code,
            venue_name=venue_name, race_no=race_no, deadline_time=deadline_str,
            top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=gatekeeper_passed,
            cluster_id=cluster_id, cluster_name=cluster_name, status=status,
            source="auto"
        )
    except Exception as e:
        logger.error(f"Supabaseへの推論結果保存エラー: {e}")

    if is_investment_go:
        total_bet = sum(bets.values())
        logger.info(f"🚀🚀🚀 [{venue_name} {race_no}R] 投資GOサイン点灯！ 推奨買い目: {len(bets)}点 / 総投資額: {total_bet:,}円")
        try:
            db_manager.save_recommended_bets(race_id, bets, benter_probs_dict, all_odds)
        except Exception as e:
            logger.error(f"Supabaseへの推奨買い目保存エラー: {e}")
        
        send_discord_notification(
            webhook_url=webhook_url,
            date_str=date_str,
            venue_code=venue_code,
            venue_name=venue_name,
            race_no=race_no,
            deadline_str=deadline_str,
            top_boat=top_boat,
            max_p1=max_p1,
            prob_gap=prob_gap,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            bets=bets,
            benter_probs=benter_probs_dict,
            all_odds=all_odds,
            bankroll=bankroll,
            dry_run=dry_run
        )
    else:
        logger.info(f"[{venue_name} {race_no}R] {status_message} (ステータス保存完了: {status})")

    return {
        'status': status,
        'top_boat': top_boat,
        'max_p1': max_p1,
        'is_investment_go': is_investment_go
    }


def settle_race_results(
    target_date: Optional[str] = None,
    source: Optional[str] = 'auto'
) -> List[Dict[str, Any]]:
    """
    未確定レースの結果を公式から取得し、推奨買い目(Sniper)および的中特化(Hit-Focused)と照合して確定損益をデータベースへ反映
    """
    if not hasattr(db_manager, 'get_unresolved_predictions'):
        try:
            import importlib
            importlib.reload(db_manager)
        except Exception:
            pass
            
    unresolved_main = db_manager.get_unresolved_predictions(date_str=target_date, source=source) if hasattr(db_manager, 'get_unresolved_predictions') else []
    unresolved_hit = db_manager.get_unresolved_hit_focused_predictions(date_str=target_date) if hasattr(db_manager, 'get_unresolved_hit_focused_predictions') else []
    unresolved_manshu = db_manager.get_unresolved_manshu_predictions(date_str=target_date) if hasattr(db_manager, 'get_unresolved_manshu_predictions') else []
    
    if not unresolved_main and not unresolved_hit and not unresolved_manshu:
        return []

    # レース結果キャッシュ (同一レースの重複スクレイピング防止)
    result_cache: Dict[Tuple[str, int, int], Optional[Dict[str, Any]]] = {}
    def fetch_result(date_s: str, v_code: int, r_no: int) -> Optional[Dict[str, Any]]:
        key = (date_s, v_code, r_no)
        if key not in result_cache:
            result_cache[key] = BoatRaceScraper.get_race_result(date_s, v_code, r_no)
        return result_cache[key]

    settled_results = []
    
    # 1. 本番・黄金ベースライン (race_predictions) の精算
    if unresolved_main:
        logger.info(f"🔍 [SETTLEMENT: MAIN] {len(unresolved_main)} 件の未確定レースの結果確認を開始します...")
        for r in unresolved_main:
            race_id = r['race_id']
            date_str = r['race_date']
            venue_code = r['venue_code']
            venue_name = r['venue_name']
            race_no = r['race_no']
            
            # モックレースの場合は結果スクレイピングをスキップ
            if "_MOCK" in race_id:
                continue
                
            res = fetch_result(date_str, venue_code, race_no)
            if not res:
                continue
                
            combo = res['combo']
            payout_per_100 = res['payout_per_100']
            
            # 推奨買い目を取得
            bets = db_manager.get_recommended_bets(race_id)
            
            if not bets:
                # 投資対象外レース (Gatekeeper見送り、難水面スキップ、EV見送り等)
                db_manager.update_race_result(
                    race_id=race_id,
                    actual_result=combo,
                    payout=0,
                    profit=0,
                    hit_status="no_bet"
                )
                settled_results.append({
                    'race_id': race_id,
                    'venue_name': venue_name,
                    'race_no': race_no,
                    'mode': 'main',
                    'actual_result': combo,
                    'hit_status': 'no_bet',
                    'payout': 0,
                    'profit': 0
                })
                continue
                
            # 投資ありレースの的中判定
            total_bet = sum(b['bet_amount'] for b in bets)
            is_cancelled = res.get('is_cancelled', False) or combo == '不成立'

            if is_cancelled:
                actual_payout = total_bet
                profit = 0
                hit_status = "refund"
                logger.info(f"🔄 [Sniper返還] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 全額返還 (収支: 0円)")
            else:
                hit_bet = next((b for b in bets if b['combination'] == combo), None)
                if hit_bet:
                    bet_amt = hit_bet['bet_amount']
                    actual_payout = int((bet_amt / 100.0) * payout_per_100)
                    profit = actual_payout - total_bet
                    hit_status = "hit"
                    logger.info(f"🎉🎉🎉 [🎯Sniper的中!] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: {actual_payout:,}円 (純利益: {profit:+,}円)")
                else:
                    actual_payout = 0
                    profit = - total_bet
                    hit_status = "miss"
                    logger.info(f"❌ [Sniper不的中] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: 0円 (損失: {profit:,}円)")
                
            db_manager.update_race_result(
                race_id=race_id,
                actual_result=combo,
                payout=actual_payout,
                profit=profit,
                hit_status=hit_status
            )
            
            settled_results.append({
                'race_id': race_id,
                'venue_name': venue_name,
                'race_no': race_no,
                'mode': 'main',
                'actual_result': combo,
                'hit_status': hit_status,
                'payout': actual_payout,
                'profit': profit
            })
            
    # 2. 的中特化 (hit_focused_predictions) の精算
    if unresolved_hit:
        logger.info(f"🔍 [SETTLEMENT: HIT-FOCUSED] {len(unresolved_hit)} 件の未確定的中特化レースの結果確認を開始します...")
        for r in unresolved_hit:
            race_id = r['race_id']
            date_str = r['race_date']
            venue_code = r['venue_code']
            venue_name = r['venue_name']
            race_no = r['race_no']
            
            # モックレースの場合は結果スクレイピングをスキップ
            if "_MOCK" in race_id:
                continue
                
            res = fetch_result(date_str, venue_code, race_no)
            if not res:
                continue
                
            combo = res['combo']
            payout_per_100 = res['payout_per_100']
            
            detail = db_manager.get_hit_focused_race_detail(race_id)
            bets = detail.get('bets', []) if detail else []
            
            if not bets:
                db_manager.update_hit_focused_result(
                    race_id=race_id,
                    actual_result=combo,
                    payout=0,
                    profit=0,
                    hit_status="no_bet"
                )
                settled_results.append({
                    'race_id': race_id,
                    'venue_name': venue_name,
                    'race_no': race_no,
                    'mode': 'hit_focused',
                    'actual_result': combo,
                    'hit_status': 'no_bet',
                    'payout': 0,
                    'profit': 0
                })
                continue
                
            total_bet = sum(b['bet_amount'] for b in bets)
            is_cancelled = res.get('is_cancelled', False) or combo == '不成立'

            if is_cancelled:
                actual_payout = total_bet
                profit = 0
                hit_status = "refund"
                logger.info(f"🔄 [的中特化 返還] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 全額返還 (収支: 0円)")
            else:
                hit_bet = next((b for b in bets if b['combination'] == combo), None)
                if hit_bet:
                    bet_amt = hit_bet['bet_amount']
                    actual_payout = int((bet_amt / 100.0) * payout_per_100)
                    profit = actual_payout - total_bet
                    hit_status = "hit"
                    logger.info(f"💮 [🎯的中特化 的中!] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: {actual_payout:,}円 (純利益: {profit:+,}円)")
                else:
                    actual_payout = 0
                    profit = - total_bet
                    hit_status = "miss"
                    logger.info(f"💀 [的中特化 不的中] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: 0円 (損失: {profit:,}円)")
                
            db_manager.update_hit_focused_result(
                race_id=race_id,
                actual_result=combo,
                payout=actual_payout,
                profit=profit,
                hit_status=hit_status
            )
            
            settled_results.append({
                'race_id': race_id,
                'venue_name': venue_name,
                'race_no': race_no,
                'mode': 'hit_focused',
                'actual_result': combo,
                'hit_status': hit_status,
                'payout': actual_payout,
                'profit': profit
            })

    # 3. 万舟スナイパー (manshu_predictions) の精算
    if unresolved_manshu:
        logger.info(f"🔍 [SETTLEMENT: MANSHU-SNIPER] {len(unresolved_manshu)} 件の未確定万舟スナイパーレースの結果確認を開始します...")
        for r in unresolved_manshu:
            race_id = r['race_id']
            date_str = r['race_date']
            venue_code = r['venue_code']
            venue_name = r['venue_name']
            race_no = r['race_no']

            if "_MOCK" in race_id:
                continue

            res = fetch_result(date_str, venue_code, race_no)
            if not res:
                continue

            combo = res['combo']
            payout_per_100 = res['payout_per_100']

            detail = db_manager.get_manshu_race_detail(race_id)
            bets = detail.get('bets', []) if detail else []

            if not bets:
                db_manager.update_manshu_result(
                    race_id=race_id,
                    actual_result=combo,
                    payout=0,
                    profit=0,
                    hit_status="no_bet"
                )
                settled_results.append({
                    'race_id': race_id,
                    'venue_name': venue_name,
                    'race_no': race_no,
                    'mode': 'manshu_sniper',
                    'actual_result': combo,
                    'hit_status': 'no_bet',
                    'payout': 0,
                    'profit': 0
                })
                continue

            total_bet = sum(b['bet_amount'] for b in bets)
            is_cancelled = res.get('is_cancelled', False) or combo == '不成立'

            if is_cancelled:
                actual_payout = total_bet
                profit = 0
                hit_status = "refund"
                logger.info(f"🔄 [万舟スナイパー 返還] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 全額返還 (収支: 0円)")
            else:
                hit_bet = next((b for b in bets if b['combination'] == combo), None)
                if hit_bet:
                    bet_amt = hit_bet['bet_amount']
                    actual_payout = int((bet_amt / 100.0) * payout_per_100)
                    profit = actual_payout - total_bet
                    hit_status = "hit"
                    logger.info(f"🎆🎆🎆 [💣万舟スナイパー 炸裂的中!!] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: {actual_payout:,}円 (純利益: {profit:+,}円)")
                else:
                    actual_payout = 0
                    profit = - total_bet
                    hit_status = "miss"
                    logger.info(f"💨 [万舟スナイパー 不的中] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: 0円 (損失: {profit:,}円)")

            db_manager.update_manshu_result(
                race_id=race_id,
                actual_result=combo,
                payout=actual_payout,
                profit=profit,
                hit_status=hit_status
            )

            settled_results.append({
                'race_id': race_id,
                'venue_name': venue_name,
                'race_no': race_no,
                'mode': 'manshu_sniper',
                'actual_result': combo,
                'hit_status': hit_status,
                'payout': actual_payout,
                'profit': profit
            })
            
    if settled_results:
        logger.info(f"🏁 [SETTLEMENT] {len(settled_results)} 件の確定収支をデータベースに反映しました。")
        
    return settled_results


def evaluate_mock_race(

    venue_code: int = 18,
    race_no: int = 10,
    bankroll: float = DEFAULT_BANKROLL,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    max_exposure: float = DEFAULT_MAX_EXPOSURE,
    max_concentration: float = DEFAULT_MAX_CONCENTRATION,
    min_ev: float = DEFAULT_MIN_EV,
    max_odds: float = DEFAULT_MAX_ODDS,
    gatekeeper_th: float = DEFAULT_GATEKEEPER_TH,
    webhook_url: str = DISCORD_WEBHOOK_URL,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    オフライン検証・開発用のリアルなモックデータによるフルパイプライン検証（DB永続化含む）
    """
    v_name = VENUE_MAP.get(venue_code, f"場{venue_code:02d}")
    date_str = datetime.date.today().strftime('%Y%m%d')
    race_id = f"{date_str}_{venue_code}_{race_no}_MOCK"
    logger.info(f"🧪 [MOCK TEST] リアルな模擬データを用いて {v_name} {race_no}R のフルパイプライン検証を実行します...")
    
    # リアルな出走表 & 展示データ生成 (1号艇が圧倒的本命)
    racers = [4320, 3960, 4444, 4012, 4500, 4210]
    motors = [52.0, 35.0, 31.0, 28.0, 29.0, 26.0]
    boats = [48.0, 34.0, 30.0, 27.0, 28.0, 25.0]
    ex_times = [6.58, 6.75, 6.80, 6.85, 6.88, 6.92]
    sts = [0.08, 0.16, 0.17, 0.19, 0.20, 0.22]
    nat_win_rates = [8.80, 6.20, 5.50, 5.00, 4.50, 4.00]
    local_win_rates = [9.10, 6.30, 5.60, 4.90, 4.40, 4.00]

    rows = []
    for i in range(6):
        bn = i + 1
        rows.append({
            'race_id': race_id,
            'boat_number': bn,
            'racer_id': racers[i],
            'motor_rate': motors[i],
            'boat_rate': boats[i],
            'exhibition_time': ex_times[i],
            'exhibition_start_timing': sts[i],
            'pred_course': bn,
            'wind_direction': 1,
            'wind_speed': 2.0,
            'wave_height': 1.0,
            'prior_results': '1 1 1 1' if bn == 1 else '3 4 2 5',
            'branch': '山口' if bn == 1 else '福岡',
            'racer_rank': 'A1' if bn == 1 else ('A2' if bn in (2, 3) else 'B1'),
            'weight': 52.0,
            'nat_win_rate': nat_win_rates[i],
            'nat_quinella_rate': 58.5 if bn == 1 else 36.0,
            'local_win_rate': local_win_rates[i]
        })
    df_race = pd.DataFrame(rows)
    
    # リアルな直前オッズ生成 (市場実態を反映しつつEV歪みを意図的に含む)
    combos = [f"{c[0]}-{c[1]}-{c[2]}" for c in itertools.permutations(range(1, 7), 3)]
    all_odds = {}
    for c in combos:
        if c == '1-2-3': all_odds[c] = 12.4
        elif c == '1-2-4': all_odds[c] = 18.2
        elif c == '1-3-2': all_odds[c] = 21.0
        elif c == '1-3-4': all_odds[c] = 24.5
        elif c == '1-2-5': all_odds[c] = 26.0
        elif c.startswith('1-2'): all_odds[c] = 14.0 + int(c[-1]) * 2.0
        elif c.startswith('1-3'): all_odds[c] = 20.0 + int(c[-1]) * 2.5
        elif c.startswith('1-'): all_odds[c] = 30.0 + int(c[-1]) * 3.0
        else: all_odds[c] = 80.0 + int(c[0]) * 20.0

    # オッズ保存
    try:
        db_manager.save_odds_batch(race_id, all_odds)
    except Exception: pass

    # 特徴量生成
    df_feat = FeatureEngineer.process(df_race, v_name, race_date=date_str)
    active_boats = df_feat['boat_number'].tolist()

    
    syn_dict = {b: 0.0 for b in active_boats}
    for combo, o_val in all_odds.items():
        try:
            b1 = int(combo.split('-')[0])
            if o_val > 0 and b1 in syn_dict:
                syn_dict[b1] += (1.0 / o_val)
        except Exception: pass
        
    total_syn = sum(syn_dict.values())
    p_norm_syn_dict = {b: syn_dict[b] / total_syn for b in active_boats} if total_syn > 0 else {b: 1.0 / 6.0 for b in active_boats}
    # model_honmei は学習データと同様に未正規化の逆数和 (Σ(1/odds)) を syn_win_rate として参照
    df_feat['syn_win_rate'] = df_feat['boat_number'].map(syn_dict).fillna(0.0)
    # model_residual のベースライン init_score には正規化後の市場勝率を使用
    p_norm_syn_series = df_feat['boat_number'].map(p_norm_syn_dict).fillna(1.0 / 6.0)
    df_feat['init_score'] = probs_to_init_scores(p_norm_syn_series.to_numpy())
    
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
    
    logger.info(f"🛡️ [MOCK] Gatekeeper P1 = {max_p1:.2%} ({top_boat}号艇本命 / 閾値: {gatekeeper_th:.2%})")
    
    b1_feat = df_feat[df_feat['boat_number'] == 1]
    b1_motor_rate = float(b1_feat['motor_rate'].iloc[0]) if (len(b1_feat) > 0 and 'motor_rate' in b1_feat.columns and pd.notna(b1_feat['motor_rate'].iloc[0])) else 0.0
    wave_height = float(df_feat['wave_height'].iloc[0]) if ('wave_height' in df_feat.columns and pd.notna(df_feat['wave_height'].iloc[0])) else 0.0

    cluster_d2, cluster_d3, cluster_id, cluster_name = get_cluster_benter_params(venue_code)
    race_info = {
        'race_name': f"{v_name} {race_no}R (MOCK)",
        'venue_name': v_name,
        'venue_code': venue_code,
        'race_no': race_no,
        'deadline': "15:25 (MOCK)",
        'cluster_id': cluster_id,
        'cluster_name': cluster_name
    }

    # Extractor 推論 (見送り時でもインメモリで全件計算)
    model_r = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
    df_r = prepare_features_for_model(df_feat, model_r)
    raw_res = model_r.predict(df_r, raw_score=True)
    total_logits = raw_res + df_feat['init_score'].to_numpy()
    p_raw_res = 1.0 / (1.0 + np.exp(-np.clip(total_logits, -30, 30)))
    p_norm_res = p_raw_res / np.sum(p_raw_res)
    p1_dict_residual = dict(zip(df_feat['boat_number'], p_norm_res))

    b1_p_res = float(p1_dict_residual.get(1, 0.0))
    b1_p_mkt = float(p_norm_syn_dict.get(1, 0.0))
    is_cluster1_venue = (venue_code in CLUSTER_1_VENUES)
    is_regime_a = (not is_cluster1_venue) and (top_boat == 1 and max_p1 >= 0.80 and b1_p_res > b1_p_mkt)
    is_regime_b = (not is_cluster1_venue) and (top_boat == 1 and b1_p_res >= 0.74 and b1_motor_rate >= 35.0)
    regime_c_heads = tuple(
        b for b in (2, 3, 4)
        if float(p1_dict_residual.get(b, 0.0)) >= 0.35
        and (float(p1_dict_residual.get(b, 0.0)) - float(p_norm_syn_dict.get(b, 0.0))) >= 0.11
    )
    is_regime_c = (len(regime_c_heads) > 0) and not (is_regime_a or is_regime_b)
    gatekeeper_passed = ((not is_cluster1_venue) and (top_boat == 1) and ((max_p1 >= gatekeeper_th) or (b1_p_res >= 0.74))) or is_regime_c

    sniper_go = False
    sniper_node = ""
    if is_regime_a:
        sniper_go = True
        sniper_node = "Regime A: 1号艇・市場過小評価オーバーレイ (Hybrid 2nd/3rd)"
    elif is_regime_b:
        sniper_go = True
        sniper_node = "Regime B: 1号艇・高信頼残差×好モーター (Benter)"
    elif is_regime_c:
        sniper_go = True
        heads_str = ",".join(f"{b}号艇" for b in regime_c_heads)
        sniper_node = f"Regime C: 逆転スナイプ [{heads_str}→1号艇2着残し] (Overlay>=11%)" 
    
    # Benter 展開
    benter_probs, _, _ = calculate_benter_probs(
        p1_dict_residual,
        d2=cluster_d2,
        d3=cluster_d3,
        calibration_method='direct'
    )
    benter_probs_dict = {p['combo']: p['prob'] for p in benter_probs}
    
    # 的中特化（動的ダッチング）をメモリ上で計算
    bets_hit_focused = {}
    exp_roi_hit = 0.0
    syn_odds_hit = 0.0
    is_gogo_lamp = False
    is_chance_lamp = False
    if benter_probs_dict and all_odds:
        try:
            bets_hit_focused = calculate_dutching_bets(
                benter_probs=benter_probs_dict,
                odds_dict=all_odds,
                budget=1000,
                target_cum_prob=0.50,
                max_combos=8,
                min_combos=2,
                min_synthetic_odds=None  # 目標合成オッズ制約を撤廃し完全な的中率特化へ
            )
            if bets_hit_focused:
                tot_hit_b = sum(bets_hit_focused.values())
                exp_pay = sum(
                    benter_probs_dict.get(c, 0.0) * all_odds.get(c, 0.0) * amt
                    for c, amt in bets_hit_focused.items()
                )
                exp_roi_hit = (exp_pay / tot_hit_b) if tot_hit_b > 0 else 0.0
                inv_sum = sum(1.0 / all_odds.get(c, 1.0) for c in bets_hit_focused.keys() if all_odds.get(c, 0) > 0)
                syn_odds_hit = 1.0 / inv_sum if inv_sum > 0 else 0.0

                # 🎰 GOGO!ランプ点灯判定 (超激アツ / Tier 1): P1>=55% & モーター2連率>=40% & 合成期待回収率>=1.15
                if (max_p1 is not None and max_p1 >= 0.55) and (b1_motor_rate >= 40.0) and (exp_roi_hit >= 1.15):
                    is_gogo_lamp = True
                # ⭐ CHANCEランプ点灯判定 (チャンス / Tier 2): P1>=55% & モーター2連率>=36% & 合成オッズ>=2.0 & 期待回収率>=1.05
                elif (max_p1 is not None and max_p1 >= 0.55) and (b1_motor_rate >= 36.0) and (syn_odds_hit >= 2.0) and (exp_roi_hit >= 1.05):
                    is_chance_lamp = True
        except Exception: pass

    # 最適化 (黄金ベースライン: 2着5・6号艇除外フィルタ適用)
    sniper_probs_dict = {
        c: (p if len(c.split('-')) == 3 and c.split('-')[1] not in ('5', '6') else 0.0)
        for c, p in benter_probs_dict.items()
    }
    optimizer = PortfolioOptimizer()
    bets = optimizer.optimize_funds(
        probabilities=sniper_probs_dict,
        odds=all_odds,
        bankroll=float(bankroll),
        risk_aversion=float(risk_aversion),
        max_exposure=max_exposure,
        max_concentration=max_concentration,
        min_ev=float(min_ev),
        max_odds=float(max_odds),
        kelly_fraction=None
    )
    
    # 判定ステータスと投資フラグの決定
    if not gatekeeper_passed:
        status = "gatekeeper_skipped"
        status_message = f"☕ [MOCK] Gatekeeper未達 (P1={max_p1:.1%} < {gatekeeper_th:.1%})"
        is_investment_go = False
    elif not sniper_go:
        status = "sniper_skipped"
        status_message = f"🎯 [MOCK] Sniper Filter見送り (P1={max_p1:.1%}, Motor={b1_motor_rate:.1f}%, Wave={wave_height:.1f}cm)"
        is_investment_go = False
    elif not bets:
        status = "no_value_bets"
        status_message = f"🔍 [MOCK] EV条件未達見送り [{sniper_node}] (EV >= {min_ev:.2f} 買い目なし)"
        is_investment_go = False
    else:
        status = "mock_investment_go"
        status_message = f"🚀 [MOCK] 投資GOサイン点灯！ [{sniper_node}] ({len(bets)}点 / {sum(bets.values()):,}円)"
        is_investment_go = True

    total_bet = sum(bets.values()) if is_investment_go else 0
    logger.info(f"{status_message} (選出買い目数: {len(bets) if is_investment_go else 0}点 / 投資総額: {total_bet:,}円)")

    export_radar_state(
        race_info=race_info,
        ai_eval={
            'p1_score': max_p1,
            'top_boat': top_boat,
            'prob_gap': prob_gap,
            'gatekeeper_passed': gatekeeper_passed,
            'motor_rate': b1_motor_rate,
            'wave_height': wave_height,
            'sniper_go': sniper_go,
            'sniper_node': sniper_node,
            'status': status,
            'status_message': status_message,
            'total_bet': total_bet,
            'bets_count': len(bets) if is_investment_go else 0,
            'is_gogo_lamp': is_gogo_lamp,
            'is_chance_lamp': is_chance_lamp,
            'expected_roi': exp_roi_hit,
            'synthetic_odds': syn_odds_hit
        },
        df_feat=df_feat,
        df_race=df_race,
        p1_dict_honmei=p1_dict_honmei,
        all_odds=all_odds,
        benter_probs_dict=benter_probs_dict,
        bets=bets if is_investment_go else {},
        bets_hit_focused=bets_hit_focused
    )

    # 的中特化データを専用DBテーブルへ自動蓄積保存
    if bets_hit_focused:
        try:
            db_manager.save_hit_focused_prediction(
                race_id=race_id,
                race_date=date_str,
                venue_code=venue_code,
                venue_name=v_name,
                race_no=race_no,
                deadline_time="15:25 (MOCK)",
                top_boat=top_boat,
                max_p1=max_p1,
                prob_gap=prob_gap,
                cluster_id=cluster_id,
                cluster_name=cluster_name,
                status=status,
                bets=bets_hit_focused,
                benter_probs=benter_probs_dict,
                all_odds=all_odds,
                target_cum_prob=0.50,
                is_gogo_lamp=is_gogo_lamp,
                is_chance_lamp=is_chance_lamp,
                expected_roi=exp_roi_hit,
                motor_rate=b1_motor_rate,
                synthetic_odds=syn_odds_hit
            )
            logger.info(f"💾 [MOCK] 的中特化データを hit_focused_predictions に自動蓄積保存しました ({len(bets_hit_focused)}点 / {sum(bets_hit_focused.values()):,}円)")
        except Exception as e:
            logger.warning(f"MOCK的中特化データの保存エラー: {e}")

    # MOCK 万舟スナイパーデータを保存
    try:
        manshu_res = calculate_manshu_sniper_bets(
            p1_dict_residual=p1_dict_residual,
            p_norm_syn_dict=p_norm_syn_dict,
            all_odds=all_odds,
            d2=0.60,
            d3=0.80,
            min_odds=100.0,
            max_odds=500.0,
            min_ev=1.20,
            max_combos=8
        )
        manshu_status = "mock_manshu_go" if manshu_res['is_manshu_go'] else "mock_manshu_skipped"
        db_manager.save_manshu_prediction(
            race_id=race_id,
            race_date=date_str,
            venue_code=venue_code,
            venue_name=v_name,
            race_no=race_no,
            deadline_time="15:25 (MOCK)",
            top_boat=manshu_res['outer_boat'] if manshu_res['is_manshu_go'] else top_boat,
            max_p1=max_p1,
            prob_gap=prob_gap,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            status=manshu_status,
            bets=manshu_res['bets'],
            benter_probs=manshu_res['probs_dict'],
            all_odds=all_odds,
            target_cum_prob=manshu_res['cum_prob'],
            is_gogo_lamp=manshu_res['is_tier1'],
            is_chance_lamp=manshu_res['is_tier2'],
            expected_roi=manshu_res['expected_roi'],
            motor_rate=b1_motor_rate,
            synthetic_odds=manshu_res['synthetic_odds'],
            b1_drop=manshu_res['b1_drop'],
            outer_ov=manshu_res['outer_ov']
        )
    except Exception as e:
        logger.warning(f"MOCK万舟スナイパーデータの保存エラー: {e}")
    
    # Supabase 保存 (推論結果・ステータス)
    mock_status = f"mock_{status}" if not status.startswith("mock_") else status
    try:
        db_manager.save_race_prediction(
            race_id=race_id, race_date=date_str, venue_code=venue_code,
            venue_name=v_name, race_no=race_no, deadline_time="15:25 (MOCK)",
            top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=gatekeeper_passed,
            cluster_id=cluster_id, cluster_name=cluster_name, status=mock_status,
            source="auto"
        )
    except Exception as e:
        logger.warning(f"MOCK推論結果のDB保存例外: {e}")

    if is_investment_go:
        try:
            db_manager.save_recommended_bets(race_id, bets, benter_probs_dict, all_odds)
        except Exception as e:
            logger.warning(f"MOCK推奨買い目のDB保存例外: {e}")
        
        send_discord_notification(
            webhook_url=webhook_url,
            date_str=date_str,
            venue_code=venue_code,
            venue_name=v_name,
            race_no=race_no,
            deadline_str="15:25 (MOCK)",
            top_boat=top_boat,
            max_p1=max_p1,
            prob_gap=prob_gap,
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            bets=bets,
            benter_probs=benter_probs_dict,
            all_odds=all_odds,
            bankroll=bankroll,
            dry_run=dry_run
        )
    else:
        logger.info(f"[MOCK] 見送りレース ({status}): ステータス保存完了 (買い目保存 & Discord送信はスキップ)")
        
    return {
        'status': status,
        'top_boat': top_boat,
        'max_p1': max_p1,
        'bets': bets if is_investment_go else {},
        'total_bet': total_bet
    }


# =====================================================================
# 5. 常駐スケジューラー & メインループ
# =====================================================================

def register_daily_schedules(
    bankroll: float = DEFAULT_BANKROLL,
    gatekeeper_th: float = DEFAULT_GATEKEEPER_TH,
    webhook_url: str = DISCORD_WEBHOOK_URL,
    dry_run: bool = False
):
    """当日の全レーススケジュールを取得し、締切5分前のトリガーを登録"""
    today_str = datetime.date.today().strftime('%Y%m%d')
    now = datetime.datetime.now()
    
    logger.info("=" * 65)
    logger.info(f"📅 [SCHEDULE MANAGER] 日次スケジュール登録開始 (日付: {today_str} / 現在時刻: {now.strftime('%H:%M:%S')})")
    logger.info("=" * 65)
    
    # 既存ジョブをクリア（毎朝リフレッシュ用）
    schedule.clear('race_eval')
    
    venues = fetch_today_venues(today_str)
    if not venues:
        logger.warning("⚠️ 本日の開催会場が見つかりませんでした。")
        return
        
    logger.info(f"🏟️ 本日の開催場数: {len(venues)} 場")
    
    registered_jobs = 0
    skipped_cluster1 = 0
    already_passed = 0
    
    # Regime C (逆転スナイプ) および 万舟スナイパー は全24会場対応のため全開催場をスケジュール登録
    # (Regime A/B および 的中特化 GOGO/CHANCE は evaluate_race 内で Cluster 1 を自動除外)
    target_venues = list(venues)
    for vcode, vname in venues:
        if vcode in CLUSTER_1_VENUES:
            logger.info(f"  ・[{vcode:02d}] {vname:<4}: 【Cluster 1 (難水面)】 Regime C / 万舟スナイパー監視対象として登録")

    from concurrent.futures import ThreadPoolExecutor, as_completed
    venue_deadlines = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_v = {executor.submit(fetch_venue_race_deadlines, today_str, vc): (vc, vn) for vc, vn in target_venues}
        for future in as_completed(future_to_v):
            vc, vn = future_to_v[future]
            try:
                venue_deadlines[(vc, vn)] = future.result()
            except Exception:
                venue_deadlines[(vc, vn)] = []
                
    for (vcode, vname), races in venue_deadlines.items():
        if not races:
            continue
            
        for r_info in races:
            rno = r_info['race_no']
            dtime_str = r_info['deadline_str']
            
            # 締切時刻 & 5分前トリガー時刻の算出
            try:
                deadline_dt = datetime.datetime.strptime(f"{today_str} {dtime_str}", "%Y%m%d %H:%M")
                trigger_dt = deadline_dt - datetime.timedelta(minutes=5)
            except Exception:
                continue
                
            if trigger_dt <= now:
                already_passed += 1
                continue
                
            trigger_time_str = trigger_dt.strftime("%H:%M")
            
            # ジョブ登録 (タグ: 'race_eval')
            schedule.every().day.at(trigger_time_str).do(
                evaluate_race,
                date_str=today_str,
                venue_code=vcode,
                venue_name=vname,
                race_no=rno,
                deadline_str=dtime_str,
                bankroll=bankroll,
                gatekeeper_th=gatekeeper_th,
                webhook_url=webhook_url,
                dry_run=dry_run
            ).tag('race_eval')
            
            registered_jobs += 1
            
    logger.info("-" * 65)
    logger.info(f"✅ スケジュール登録完了: {registered_jobs} レースを待機キューに登録")
    logger.info(f"   (難水面スキップ: {skipped_cluster1}場, 発走済みスキップ: {already_passed}レース)")
    logger.info("=" * 65 + "\n")


def run_worker_loop(
    bankroll: float = DEFAULT_BANKROLL,
    gatekeeper_th: float = DEFAULT_GATEKEEPER_TH,
    webhook_url: str = DISCORD_WEBHOOK_URL,
    dry_run: bool = False
):
    """常駐ループ実行"""
    logger.info("🚀 ボートレース AI 自動通知ワーカー (`auto_trader.py`) を起動しました。")
    if webhook_url:
        logger.info("🔗 Discord Webhook: 接続設定済み")
    else:
        logger.warning("⚠️ Discord Webhook: 未設定 (ドライラン表示のみ行います)")
        
    # データベースの初期化 & マイグレーション確認 & 展示タイム同期
    try:
        db_manager.init_database()
        db_manager.sync_recent_exhibitions(days=10)
    except Exception as e:
        logger.error(f"データベース初期化 / 展示同期エラー: {e}")
        
    # 起動時に即座に当日のスケジュールを登録 & 過去レース結果を精算
    register_daily_schedules(
        bankroll=bankroll,
        gatekeeper_th=gatekeeper_th,
        webhook_url=webhook_url,
        dry_run=dry_run
    )
    try:
        settle_race_results()
    except Exception as e:
        logger.warning(f"起動時レース結果精算例外: {e}")
    
    # 毎朝 08:00 に翌日/当日の全場スケジュールを自動リフレッシュ登録 & 展示タイム同期
    schedule.every().day.at("08:00").do(
        register_daily_schedules,
        bankroll=bankroll,
        gatekeeper_th=gatekeeper_th,
        webhook_url=webhook_url,
        dry_run=dry_run
    )
    schedule.every().day.at("08:00").do(
        db_manager.sync_recent_exhibitions,
        days=10
    ).tag('exhibition_sync')
    
    # 毎週月曜 03:00 (夜間アイドル時) に選手・コース外部マスタデータを自動更新
    if update_all_masters is not None:
        def safe_update_masters():
            try:
                logger.info("🔄 [週次バッチ] 選手・コース外部マスタテーブルの自動更新を実行します...")
                res = update_all_masters()
                logger.info(f"✅ [週次バッチ] 外部マスタ更新完了: {res}")
            except Exception as ex:
                logger.error(f"⚠️ [週次バッチ] 外部マスタ更新例外: {ex}", exc_info=True)
        schedule.every().monday.at("03:00").do(safe_update_masters).tag('weekly_master_update')
        logger.info("📅 週次マスタ自動更新ジョブを登録しました: 毎週月曜 03:00 実行")

    # 10分ごとに未確定レースの結果を取得・的中判定・収支更新
    schedule.every(10).minutes.do(settle_race_results).tag('settlement')
    
    logger.info("⏳ 待機ループ開始... (Ctrl+C で停止)")
    try:
        while True:
            try:
                schedule.run_pending()
            except Exception as e:
                logger.error(f"⚠️ スケジューラージョブ実行中に例外が発生しました（監視ループを継続します）: {e}", exc_info=True)
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("🛑 ワーカーを正常に停止しました。")


# =====================================================================
# 6. CLI エントリーポイント
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BOATRACE AI Auto Trader Worker")
    parser.add_argument('--test', action='store_true', help="Run single race test evaluation immediately")
    parser.add_argument('--mock', action='store_true', help="Run mock evaluation without web requests (ideal for midnight/offline tests)")
    parser.add_argument('--settle', action='store_true', help="Run race result settlement & payout calculation for unresolved races")
    parser.add_argument('--sync', action='store_true', help="Sync recent 10 days exhibition times to Supabase (recent_exhibitions)")
    parser.add_argument('--date', type=str, default='', help="Date for test in YYYYMMDD (default: today)")
    parser.add_argument('--venue', type=int, default=18, help="Venue code for test (1-24, default: 18/Tokuyama)")
    parser.add_argument('--race', type=int, default=10, help="Race number for test (1-12, default: 10)")
    parser.add_argument('--bankroll', type=float, default=DEFAULT_BANKROLL, help="Simulated bankroll (default: 100,000)")
    parser.add_argument('--min_ev', type=float, default=DEFAULT_MIN_EV, help="Minimum EV threshold (default: 1.25)")
    parser.add_argument('--max_odds', type=float, default=DEFAULT_MAX_ODDS, help="Maximum Odds threshold (default: 30.0)")
    parser.add_argument('--p1_th', type=float, default=DEFAULT_GATEKEEPER_TH, help="Gatekeeper P1 threshold (default: 0.7438)")
    parser.add_argument('--dry-run', action='store_true', help="Run without sending Discord webhook (console output only)")
    parser.add_argument('--update-masters', action='store_true', help="Update static master tables in app_data/ from boatrace.db immediately")
    
    args = parser.parse_args()
    
    # DB初期化
    try:
        db_manager.init_database()
    except Exception as e:
        logger.warning(f"DB初期化スキップ: {e}")
        
    webhook = DISCORD_WEBHOOK_URL
    if args.dry_run:
        webhook = ''
        
    if args.update_masters:
        if update_all_masters is not None:
            logger.info("🔄 外部マスタテーブルの手動即時更新を実行します...")
            res = update_all_masters()
            print(f"✅ 外部マスタ更新完了: {res}")
        else:
            print("❌ update_static_masters モジュールがロードできませんでした。")
    elif args.sync:
        # 展示タイム同期モード
        logger.info("🔄 直近10日分の展示タイムを Supabase へ同期します...")
        synced = db_manager.sync_recent_exhibitions(days=10)
        print(f"✅ 展示タイム同期完了: {synced} 件のレコードを Supabase (recent_exhibitions) へ同期しました。")
    elif args.settle:

        # 結果精算・確定収支計算モード
        target_d = args.date if args.date else None
        logger.info(f"🏁 未確定レースの結果取得・確定収支計算を実行します (Date: {target_d or 'All'})...")
        settled = settle_race_results(target_date=target_d)
        print(f"✅ 精算完了: {len(settled)} 件のレース結果を更新しました。")
    elif args.mock:
        # モック検証モード
        evaluate_mock_race(
            venue_code=args.venue,
            race_no=args.race,
            bankroll=args.bankroll,
            min_ev=args.min_ev,
            max_odds=args.max_odds,
            gatekeeper_th=args.p1_th,
            webhook_url=webhook,
            dry_run=args.dry_run
        )
    elif args.test:
        test_date = args.date if args.date else datetime.date.today().strftime('%Y%m%d')
        v_name = VENUE_MAP.get(args.venue, f"場{args.venue:02d}")
        
        # 開催場チェック
        today_venues = fetch_today_venues(test_date)
        holding_codes = [v[0] for v in today_venues]
        
        if holding_codes and args.venue not in holding_codes:
            holding_names = ", ".join([f"{v[1]}({v[0]:02d})" for v in today_venues])
            logger.warning(
                f"⚠️ 指定された日付 ({test_date}) は 【{v_name}】 ではレースが開催されていません。\n"
                f"🏟️ 本日の開催場: {holding_names}\n"
                f"💡 パイプライン全体の動作確認を行いたい場合は '--mock' オプションを付与してください。\n"
                f"   例: python auto_trader.py --mock --dry-run"
            )
        else:
            logger.info(f"🧪 [TEST MODE] 実レースデータ取得テスト: {test_date} {v_name} {args.race}R")
            res = evaluate_race(
                date_str=test_date,
                venue_code=args.venue,
                venue_name=v_name,
                race_no=args.race,
                deadline_str="テスト時刻",
                bankroll=args.bankroll,
                min_ev=args.min_ev,
                max_odds=args.max_odds,
                gatekeeper_th=args.p1_th,
                webhook_url=webhook,
                dry_run=args.dry_run
            )
            if res.get('status') in ('error_data', 'error_odds'):
                logger.info(
                    f"💡 レース未開催または展示開始前の時間帯です。\n"
                    f"   今すぐ推論・最適化・通知レイアウトの動作確認を行いたい場合は '--mock' を付与してください。\n"
                    f"   例: python auto_trader.py --mock --dry-run"
                )
    else:
        run_worker_loop(
            bankroll=args.bankroll,
            gatekeeper_th=args.p1_th,
            webhook_url=webhook,
            dry_run=args.dry_run
        )

