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

# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AutoTrader")

# 定数 & パス設定
MODEL_HONMEI_PATH = 'model_honmei.txt'
MODEL_RESIDUAL_PATH = 'model_residual.txt'
DATA_DIR = 'app_data'
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
    def get_soup(cls, url: str, timeout: int = 20, max_retries: int = 3) -> Optional[BeautifulSoup]:
        session = cls.get_session()
        for attempt in range(max_retries):
            try:
                resp = session.get(url, timeout=timeout)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or 'utf-8'
                return BeautifulSoup(resp.text, 'html.parser')
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.debug(f"データ取得リトライ ({attempt+1}/{max_retries}) [{url}]: {e}")
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
                                'payout_per_100': payout
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
            # 展示タイムパース
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


def fetch_series_momentum(venue_code: int, racer_ids: list, race_date: str = None) -> dict:
    """
    対象会場・今節（同一会場で過去7日以内〜当日）における出走選手の過去展示タイム履歴をDBから取得し、
    前走展示タイム差 (ex_momentum_diff) および 節間平均偏差 (ex_momentum_deviation) を算出する。
    """
    momentum_dict = {rid: {'ex_momentum_diff': 0.0, 'ex_momentum_deviation': 0.0} for rid in racer_ids}
    if not racer_ids:
        return momentum_dict

    sqlite_paths = ['boatrace.db', r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db']
    sqlite_db = None
    for sp in sqlite_paths:
        if os.path.exists(sp):
            sqlite_db = sp
            break

    try:
        if sqlite_db:
            import sqlite3
            conn = sqlite3.connect(sqlite_db)
            cursor = conn.cursor()
            placeholders = ','.join(['?'] * len(racer_ids))
            
            if race_date:
                date_filter = "AND r.race_date <= ?"
                params = [venue_code] + list(racer_ids) + [race_date]
            else:
                date_filter = ""
                params = [venue_code] + list(racer_ids)

            query = f"""
            SELECT re.racer_id, bi.exhibition_time
            FROM before_info bi
            JOIN races r ON bi.race_id = r.race_id
            JOIN race_entries re ON bi.race_id = re.race_id AND bi.boat_number = re.boat_number
            WHERE r.venue_code = ?
              AND re.racer_id IN ({placeholders})
              {date_filter}
              AND bi.exhibition_time > 0
            ORDER BY r.race_date ASC, r.race_number ASC
            """
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            racer_times = {}
            for r_id, ex_time in rows:
                r_id = int(r_id)
                if r_id not in racer_times:
                    racer_times[r_id] = []
                racer_times[r_id].append(float(ex_time))
                
            for r_id in racer_ids:
                times = racer_times.get(r_id, [])
                if len(times) >= 2:
                    current_ex = times[-1]
                    prev_ex = times[-2]
                    exp_mean = np.mean(times)
                    momentum_dict[r_id]['ex_momentum_diff'] = float(current_ex - prev_ex)
                    momentum_dict[r_id]['ex_momentum_deviation'] = float(current_ex - exp_mean)
                elif len(times) == 1:
                    momentum_dict[r_id]['ex_momentum_diff'] = 0.0
                    momentum_dict[r_id]['ex_momentum_deviation'] = 0.0
    except Exception as e:
        logger.debug(f"fetch_series_momentum error: {e}")
        
    return momentum_dict



class FeatureEngineer:
    @staticmethod
    def process(df: pd.DataFrame, venue_name: str, race_date: str = None) -> pd.DataFrame:
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

            df = df.merge(v_course, left_on=['venue_name', 'pred_course'], right_on=['venue_name', 'course_number'], how='left')
            df.rename(columns={'rate_1st': 'venue_course_1st_rate', 'rate_2nd': 'venue_course_2nd_rate', 'rate_3rd': 'venue_course_3rd_rate'}, inplace=True)
            df = df.merge(r_params, on='racer_id', how='left')
        except Exception: pass
        
        required_cols = ['makuri_count', 'nige_count', 'sashi_count', 'makurizashi_count', 'nat_win_rate', 'course_run_count', 'local_win_rate']
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
        denom = df['course_run_count'].replace(0, 1)
        df['makuri_rate'] = df['makuri_count'] / denom
        df['nige_rate'] = df['nige_count'] / denom
        df['sashi_rate'] = df.get('sashi_count', 0.0) / denom
        df['makurizashi_rate'] = df.get('makurizashi_count', 0.0) / denom

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
        
        min_t = df['exhibition_time'].min()
        mean_t = df['exhibition_time'].mean()
        std_t = df['exhibition_time'].std()
        if std_t == 0 or np.isnan(std_t): std_t = 1.0
        df['tenji_z_score'] = (mean_t - df['exhibition_time']) / std_t
        df['linear_rank'] = df['exhibition_time'].rank(method='min', ascending=True)
        df['is_linear_leader'] = (df['linear_rank'] == 1).astype(int)

        # 新規代替モメンタム & レース内展示偏差
        df['ex_diff_from_race_min'] = (df['exhibition_time'] - min_t).fillna(0.0)
        df['ex_diff_from_race_mean'] = (df['exhibition_time'] - mean_t).fillna(0.0)
        df['ex_rank_in_race'] = df['linear_rank']
        
        # 節間展示タイムモメンタム（動的DBクエリ結果をマッピング）
        racer_ids_list = df['racer_id'].dropna().astype(int).tolist()
        v_code_val = int(df['venue_code_int'].iloc[0]) if 'venue_code_int' in df.columns and len(df) > 0 else 1
        momentum_map = fetch_series_momentum(v_code_val, racer_ids_list, race_date)
        
        df['ex_momentum_diff'] = df['racer_id'].map(lambda r: momentum_map.get(int(r), {}).get('ex_momentum_diff', 0.0)).fillna(0.0)
        df['ex_momentum_deviation'] = df['racer_id'].map(lambda r: momentum_map.get(int(r), {}).get('ex_momentum_deviation', 0.0)).fillna(0.0)
        
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
        df['strong_wind_outer_adv'] = df['is_strong_wind'] * (df['boat_number'] >= 3).astype(float)
        df['wind_nige_vulnerability'] = df['wind_speed'] * (1.0 - df['nige_rate']) * (df['boat_number'] == 1).astype(float)


        df['wave_weight_prod'] = df['wave_height'] * df['weight']
        df['wave_weight_ratio'] = df['wave_height'] / np.maximum(df['weight'], 40.0)
        df['is_high_wave'] = (df['wave_height'] >= 4.0).astype(float)
        df['high_wave_heavy_penalty'] = df['is_high_wave'] * np.maximum(0.0, df['weight'] - 52.0)
        df['high_wave_inner_risk'] = df['is_high_wave'] * (df['boat_number'] == 1).astype(float)
        
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
# 4. 黄金ベースライン レース評価エンジン
# =====================================================================

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
    logger.info(f"🔍 [EVALUATING] {venue_name} {race_no}R (締切: {deadline_str}) の分析を開始...")
    
    # 1. 難水面（Cluster 1）即時除外ガード
    if venue_code in CLUSTER_1_VENUES:
        logger.info(f"🛑 [SKIP] {venue_name} (会場コード: {venue_code:02d}) は難水面 (Cluster 1) のためシステム保護によりスキップします。")
        try:
            db_manager.save_race_prediction(
                race_id=race_id, race_date=date_str, venue_code=venue_code,
                venue_name=venue_name, race_no=race_no, deadline_time=deadline_str,
                top_boat=None, max_p1=None, prob_gap=None, gatekeeper_passed=False,
                cluster_id=1, cluster_name="難水面・波乱場", status="skipped_cluster1",
                source="auto"
            )

        except Exception: pass
        return {'status': 'skipped_cluster1'}
        
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
        
    df_feat['syn_win_rate'] = df_feat['boat_number'].map(p_norm_syn_dict).fillna(1.0 / len(active_boats))
    df_feat['init_score'] = probs_to_init_scores(df_feat['syn_win_rate'].to_numpy())
    
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
    
    cluster_d2, cluster_d3, cluster_id, cluster_name = get_cluster_benter_params(venue_code)
    
    logger.info(f"🛡️ [{venue_name} {race_no}R] Gatekeeper P1 = {max_p1:.2%} ({top_boat}号艇本命 / 閾値: {gatekeeper_th:.2%})")
    
    # Gatekeeper 閾値判定 ($P_1 \ge 0.7438$)
    if max_p1 < gatekeeper_th:
        logger.info(f"☕ [{venue_name} {race_no}R] Gatekeeper 未達 (P1 = {max_p1:.2%} < {gatekeeper_th:.2%}) -> 見送り (No Bet)")
        try:
            db_manager.save_race_prediction(
                race_id=race_id, race_date=date_str, venue_code=venue_code,
                venue_name=venue_name, race_no=race_no, deadline_time=deadline_str,
                top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=False,
                cluster_id=cluster_id, cluster_name=cluster_name, status="gatekeeper_skipped",
                source="auto"
            )
        except Exception: pass
        return {
            'status': 'gatekeeper_skipped',
            'top_boat': top_boat,
            'max_p1': max_p1
        }
        
    logger.info(f"🎯 [{venue_name} {race_no}R] Gatekeeper 通過！ (P1 = {max_p1:.2%} ≧ {gatekeeper_th:.2%}) Extractor推論を実行します...")
    
    # 6. Extractor 推論 (Residual)
    model_r = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
    df_r = prepare_features_for_model(df_feat, model_r)
    raw_res = model_r.predict(df_r, raw_score=True)
    total_logits = raw_res + df_feat['init_score'].to_numpy()
    p_raw_res = 1.0 / (1.0 + np.exp(-np.clip(total_logits, -30, 30)))
    p_norm_res = p_raw_res / np.sum(p_raw_res)
    p1_dict_residual = dict(zip(df_feat['boat_number'], p_norm_res))
    
    # 7. 会場クラスタ別 Benter 確率展開
    benter_probs, _, _ = calculate_benter_probs(
        p1_dict_residual,
        d2=cluster_d2,
        d3=cluster_d3,
        calibration_method='direct'
    )
    benter_probs_dict = {p['combo']: p['prob'] for p in benter_probs}
    
    # 8. ポートフォリオ最適化 (SLSQP / 固定ウェイト 5%)
    optimizer = PortfolioOptimizer()
    bets = optimizer.optimize_funds(
        probabilities=benter_probs_dict,
        odds=all_odds,
        bankroll=float(bankroll),
        risk_aversion=float(risk_aversion),
        max_exposure=max_exposure,
        max_concentration=max_concentration,
        min_ev=float(min_ev),
        max_odds=float(max_odds),
        kelly_fraction=None  # 固定ウェイト (レース上限5%, 買い目上限2%)
    )
    
    # 9. 結果判定 & Supabase永続化 & Discord通知
    if not bets:
        logger.info(f"🔍 [{venue_name} {race_no}R] Gatekeeper通過も、条件を満たすEV買い目なし (EV >= {min_ev:.2f}, Odds <= {max_odds:.1f}) -> 見送り")
        try:
            db_manager.save_race_prediction(
                race_id=race_id, race_date=date_str, venue_code=venue_code,
                venue_name=venue_name, race_no=race_no, deadline_time=deadline_str,
                top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=True,
                cluster_id=cluster_id, cluster_name=cluster_name, status="no_value_bets",
                source="auto"
            )
        except Exception: pass
        return {
            'status': 'no_value_bets',
            'top_boat': top_boat,
            'max_p1': max_p1
        }
        
    total_bet = sum(bets.values())
    logger.info(f"🚀🚀🚀 [{venue_name} {race_no}R] 投資GOサイン点灯！ 推奨買い目: {len(bets)}点 / 総投資額: {total_bet:,}円")
    
    # Supabase へ推論結果および買い目を永続化
    try:
        db_manager.save_race_prediction(
            race_id=race_id, race_date=date_str, venue_code=venue_code,
            venue_name=venue_name, race_no=race_no, deadline_time=deadline_str,
            top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=True,
            cluster_id=cluster_id, cluster_name=cluster_name, status="investment_go",
            source="auto"
        )
        db_manager.save_recommended_bets(race_id, bets, benter_probs_dict, all_odds)

    except Exception as e:
        logger.error(f"Supabaseへの推論結果保存エラー: {e}")
    
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
    
    return {
        'status': 'investment_go',
        'top_boat': top_boat,
        'max_p1': max_p1,
        'bets': bets,
        'total_bet': total_bet
    }


def settle_race_results(
    target_date: Optional[str] = None,
    source: Optional[str] = 'auto'
) -> List[Dict[str, Any]]:
    """
    未確定レースの結果を公式から取得し、推奨買い目と照合して確定損益をSupabaseへ反映
    """
    if not hasattr(db_manager, 'get_unresolved_predictions'):
        try:
            import importlib
            importlib.reload(db_manager)
        except Exception:
            pass
            
    unresolved = db_manager.get_unresolved_predictions(date_str=target_date, source=source) if hasattr(db_manager, 'get_unresolved_predictions') else []
    if not unresolved:
        return []

        
    logger.info(f"🔍 [SETTLEMENT] {len(unresolved)} 件の未確定レースの結果確認を開始します...")
    settled_results = []
    
    for r in unresolved:
        race_id = r['race_id']
        date_str = r['race_date']
        venue_code = r['venue_code']
        venue_name = r['venue_name']
        race_no = r['race_no']
        status = r['status']
        
        # モックレースの場合は結果スクレイピングをスキップ
        if "_MOCK" in race_id:
            continue
            
        result = BoatRaceScraper.get_race_result(date_str, venue_code, race_no)
        if not result:
            continue
            
        combo = result['combo']
        payout_per_100 = result['payout_per_100']
        
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
                'actual_result': combo,
                'hit_status': 'no_bet',
                'payout': 0,
                'profit': 0
            })
            continue
            
        # 投資ありレースの的中判定
        total_bet = sum(b['bet_amount'] for b in bets)
        hit_bet = next((b for b in bets if b['combination'] == combo), None)
        
        if hit_bet:
            bet_amt = hit_bet['bet_amount']
            actual_payout = int((bet_amt / 100.0) * payout_per_100)
            profit = actual_payout - total_bet
            hit_status = "hit"
            logger.info(f"🎉🎉🎉 [🎯的中!] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: {actual_payout:,}円 (純利益: {profit:+,}円)")
        else:
            actual_payout = 0
            profit = - total_bet
            hit_status = "miss"
            logger.info(f"❌ [不的中] {venue_name} {race_no}R: 結果 {combo} | 投資: {total_bet:,}円 -> 払戻: 0円 (損失: {profit:,}円)")
            
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
            'actual_result': combo,
            'hit_status': hit_status,
            'payout': actual_payout,
            'profit': profit
        })
        
    if settled_results:
        logger.info(f"🏁 [SETTLEMENT] {len(settled_results)} 件のレース確定収支をデータベースに反映しました。")
        
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
            'weight': 52.0,
            'nat_win_rate': nat_win_rates[i],
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
    p_norm_syn_dict = {b: syn_dict[b] / total_syn for b in active_boats}
    df_feat['syn_win_rate'] = df_feat['boat_number'].map(p_norm_syn_dict).fillna(1.0 / 6.0)
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
    
    logger.info(f"🛡️ [MOCK] Gatekeeper P1 = {max_p1:.2%} ({top_boat}号艇本命 / 閾値: {gatekeeper_th:.2%})")
    
    # Extractor 推論
    model_r = lgb.Booster(model_file=MODEL_RESIDUAL_PATH)
    df_r = prepare_features_for_model(df_feat, model_r)
    raw_res = model_r.predict(df_r, raw_score=True)
    total_logits = raw_res + df_feat['init_score'].to_numpy()
    p_raw_res = 1.0 / (1.0 + np.exp(-np.clip(total_logits, -30, 30)))
    p_norm_res = p_raw_res / np.sum(p_raw_res)
    p1_dict_residual = dict(zip(df_feat['boat_number'], p_norm_res))
    
    # Benter 展開
    cluster_d2, cluster_d3, cluster_id, cluster_name = get_cluster_benter_params(venue_code)
    benter_probs, _, _ = calculate_benter_probs(
        p1_dict_residual,
        d2=cluster_d2,
        d3=cluster_d3,
        calibration_method='direct'
    )
    benter_probs_dict = {p['combo']: p['prob'] for p in benter_probs}
    
    # 最適化
    optimizer = PortfolioOptimizer()
    bets = optimizer.optimize_funds(
        probabilities=benter_probs_dict,
        odds=all_odds,
        bankroll=float(bankroll),
        risk_aversion=float(risk_aversion),
        max_exposure=max_exposure,
        max_concentration=max_concentration,
        min_ev=float(min_ev),
        max_odds=float(max_odds),
        kelly_fraction=None
    )
    
    total_bet = sum(bets.values()) if bets else 0
    logger.info(f"🚀 [MOCK] 最適化選出買い目数: {len(bets)}点 / 投資総額: {total_bet:,}円")
    
    # Supabase 保存
    try:
        db_manager.save_race_prediction(
            race_id=race_id, race_date=date_str, venue_code=venue_code,
            venue_name=v_name, race_no=race_no, deadline_time="15:25 (MOCK)",
            top_boat=top_boat, max_p1=max_p1, prob_gap=prob_gap, gatekeeper_passed=True,
            cluster_id=cluster_id, cluster_name=cluster_name, status="mock_investment_go",
            source="auto"
        )

        if bets:
            db_manager.save_recommended_bets(race_id, bets, benter_probs_dict, all_odds)
    except Exception as e:
        logger.warning(f"MOCK推論結果のDB保存例外: {e}")
    
    if bets:
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
        logger.info("[MOCK] 最適化条件を満たす買い目が選出されなかったため、Discord送信は見送られました。")
        
    return {
        'status': 'mock_success',
        'top_boat': top_boat,
        'max_p1': max_p1,
        'bets': bets,
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
    
    for vcode, vname in venues:
        if vcode in CLUSTER_1_VENUES:
            skipped_cluster1 += 1
            logger.info(f"  ・[{vcode:02d}] {vname:<4}: 【難水面 (Cluster 1) 除外】 登録スキップ")
            continue
            
        races = fetch_venue_race_deadlines(today_str, vcode)
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
        
    # データベースの初期化 & マイグレーション確認
    try:
        db_manager.init_database()
    except Exception as e:
        logger.error(f"データベース初期化エラー: {e}")
        
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
    
    # 毎朝 08:00 に翌日/当日の全場スケジュールを自動リフレッシュ登録
    schedule.every().day.at("08:00").do(
        register_daily_schedules,
        bankroll=bankroll,
        gatekeeper_th=gatekeeper_th,
        webhook_url=webhook_url,
        dry_run=dry_run
    )
    
    # 10分ごとに未確定レースの結果を取得・的中判定・収支更新
    schedule.every(10).minutes.do(settle_race_results).tag('settlement')
    
    logger.info("⏳ 待機ループ開始... (Ctrl+C で停止)")
    try:
        while True:
            schedule.run_pending()
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
    parser.add_argument('--date', type=str, default='', help="Date for test in YYYYMMDD (default: today)")
    parser.add_argument('--venue', type=int, default=18, help="Venue code for test (1-24, default: 18/Tokuyama)")
    parser.add_argument('--race', type=int, default=10, help="Race number for test (1-12, default: 10)")
    parser.add_argument('--bankroll', type=float, default=DEFAULT_BANKROLL, help="Simulated bankroll (default: 100,000)")
    parser.add_argument('--min_ev', type=float, default=DEFAULT_MIN_EV, help="Minimum EV threshold (default: 1.25)")
    parser.add_argument('--max_odds', type=float, default=DEFAULT_MAX_ODDS, help="Maximum Odds threshold (default: 30.0)")
    parser.add_argument('--p1_th', type=float, default=DEFAULT_GATEKEEPER_TH, help="Gatekeeper P1 threshold (default: 0.7438)")
    parser.add_argument('--dry-run', action='store_true', help="Run without sending Discord webhook (console output only)")
    
    args = parser.parse_args()
    
    # DB初期化
    try:
        db_manager.init_database()
    except Exception as e:
        logger.warning(f"DB初期化スキップ: {e}")
        
    webhook = DISCORD_WEBHOOK_URL
    if args.dry_run:
        webhook = ''
        
    if args.settle:
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

