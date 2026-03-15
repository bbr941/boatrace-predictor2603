# src/data_fetcher.py
import asyncio
import aiohttp
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import pandas as pd
import numpy as np
import time
import re
import logging
import itertools
import traceback
import warnings
import unicodedata

import unicodedata

# XMLParsedAsHTMLWarning を無視する設定
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
# ロガーの設定
log_format = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
main_logger = logging.getLogger(__name__)

# config モジュールを正しくインポート
try:
    from . import config
except ImportError:
    import config # スクリプトとして直接実行する場合のフォールバック

# --- HTML取得 ---
async def _fetch_url(url: str) -> str | None:
    """非同期でURLからHTMLを取得"""
    main_logger.debug(f"  Fetching: {url}")
    await asyncio.sleep(config.FETCH_SLEEP_SECONDS)
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=config.HEADERS, timeout=30) as response:
                # Correctly indented block starts here
                response.raise_for_status()
                main_logger.debug(f"  取得成功 (Status: {response.status})")
                encoding = response.charset or 'utf-8'
                html_text = await response.text(encoding=encoding)
                if not html_text or "<html" not in html_text.lower():
                    main_logger.warning(f"  取得したHTMLが空または不正な可能性があります。 URL: {url}")
                return html_text
    except asyncio.TimeoutError:
        main_logger.warning(f"  タイムアウトエラー: {url}")
        return None
    except aiohttp.ClientResponseError as http_err:
        main_logger.warning(f"  HTTPエラー {http_err.status}: {url} - {http_err.message}")
        return None
    except aiohttp.ClientError as client_err:
        main_logger.warning(f"  クライアントエラー ({url}): {client_err}")
        return None
    except Exception as e:
        main_logger.error(f"  URL取得中に予期せぬエラー ({url}): {e}")
        main_logger.error(traceback.format_exc())
        return None

# --- ヘルパー関数: グレード推定 ---
def get_race_grade(title: str | None, grade_class: str | None) -> int:
    """レースタイトルとクラス情報からグレードを推定して数値化"""
    if title is None: title = ""
    normalized_title = title.replace('　','').replace(' ','')
    if grade_class:
        if 'is-sg' in grade_class: return 6
        if 'is-g1' in grade_class: return 5
        if 'is-g2' in grade_class: return 4
        if 'is-g3' in grade_class: return 3
        if 'is-ippan' in grade_class: return 1
    if any(kw in normalized_title for kw in ["グランプリ", "ＧＰ", "クラシック", "オールスター", "メモリアル", "ダービー", "オーシャンカップ", "チャレンジカップ", "ＳＧ"]): return 6
    if any(kw in normalized_title for kw in ["周年", "ダイヤモンドカップ", "レディースチャンピオン", "ヤングダービー", "マスターズチャンピオン", "クイーンズクライマックス", "高松宮", "地区選手権", "チャンピオンカップ", "キングカップ", "ウェイキーカップ", "つつじ賞", "トコタンキング", "センプルカップ", "太閤賞", "近松賞", "海の王者", "福岡チャンピオン", "全日本覇者", "海の女王", "ＧⅠ", "Ｇ１"]): return 5
    if any(kw in normalized_title for kw in ["レディースオールスター", "モーターボート大賞", "ＭＢ大賞", "全国ボートレース甲子園", "ＧⅡ", "Ｇ２"]): return 4
    if any(kw in normalized_title for kw in ["企業杯", "オールレディース", "イースタンヤング", "ウエスタンヤング", "レディースカップ", "アクアクイーン", "酒蔵杯", "ビール杯", "キリンカップ", "アサヒビール", "ＧⅢ", "Ｇ３"]): return 3
    if '一般' in normalized_title: return 1
    return 1

# --- ヘルパー関数: ナイター判定 ---
def get_is_night_race(title: str | None) -> int:
    """レースタイトルからナイターレースか判定"""
    if title is None: return 0
    return 1 if re.search('ナイター|ナイト', title, re.IGNORECASE) else 0

# --- ヘルパー関数: 開催日計算 ---
def get_tournament_day(day_str: str | None) -> int:
    """開催日文字列 ('初日', '２日目'等) を数値に変換"""
    if day_str is None: return 0
    day_map = {'初': 1, '１': 1, '1': 1, '２': 2, '2': 2, '３': 3, '3': 3, '４': 4, '4': 4,
               '５': 5, '5': 5, '６': 6, '6': 6, '７': 7, '7': 7, '最終': 99, '優 勝': 99}
    match = re.search(r'(\d+|[０-９]+|初|最終|優\s*勝)', day_str)
    if match:
        key = match.group(1).replace(' ','')
        key = key.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
        return day_map.get(key, 0)
    return 0

# --- レースヘッダー情報解析関数 ---
def _parse_race_header_info(html_content: str) -> dict:
    """出走表HTMLからレースタイトル、開催日などを解析"""
    header_info = {'race_title': None, 'day_num_str': None, 'race_grade_num': 0, 'is_night_race': 0, 'tournament_day': 0}
    if not html_content:
        main_logger.warning("レースヘッダー情報解析: HTMLコンテンツが空です。")
        return header_info
    main_logger.debug("レースヘッダー情報の解析開始...")
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        title_h2 = soup.select_one("div.heading2_title h2.heading2_titleName")
        if title_h2:
            header_info['race_title'] = title_h2.get_text(strip=True)
            title_div = title_h2.parent
            grade_class_str = " ".join(title_div.get('class', [])) if title_div else ""
            header_info['race_grade_num'] = get_race_grade(header_info['race_title'], grade_class_str)
            header_info['is_night_race'] = get_is_night_race(header_info['race_title'])
        else: main_logger.warning("レースタイトル要素が見つかりません。")
        active_day_span = soup.select_one("div.tab2 li.is-active2 span.tab2_inner span")
        if active_day_span:
            header_info['day_num_str'] = active_day_span.get_text(strip=True)
            header_info['tournament_day'] = get_tournament_day(header_info['day_num_str'])
        else: main_logger.warning("開催日要素が見つかりません。")
    except Exception as e:
        main_logger.error(f"レースヘッダー情報解析中にエラーが発生: {e}", exc_info=True)
    main_logger.debug("レースヘッダー情報の解析完了。")
    return header_info

# --- 出走表 解析 ---
def _parse_racecard(html_content: str) -> pd.DataFrame | None:
    """出走表HTML解析"""
    if not html_content:
        main_logger.warning("  エラー: 出走表HTMLコンテンツが空です。")
        return None
    main_logger.info("  出走表HTMLを解析中...")
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        entries = []
        tbody_elements = soup.select("tbody.is-fs12")

        if not tbody_elements:
             main_logger.warning("  出走者情報を含むtbody要素(tbody.is-fs12)が見つかりません。")
             return None

        main_logger.debug(f"  検出された出走者ブロック(tbody)数: {len(tbody_elements)}")

        for boat_idx, tbody in enumerate(tbody_elements, 0):
            boat_number = boat_idx + 1
            entry = {'boat_number': boat_number}
            main_logger.debug(f"--- 艇 {boat_number} の解析開始 ---")

            try:
                boat_color_cell = tbody.select_one(f"td.is-boatColor{boat_number}")
                if not boat_color_cell:
                    main_logger.debug(f"  艇{boat_number}: 枠番セル(td.is-boatColor{boat_number})が見つかりません。")

                info_cells = tbody.select("td[rowspan='4']")
                main_logger.debug(f"  艇{boat_number}: 取得した主要情報セル(rowspan=4)の数: {len(info_cells)}")

                player_info_idx = 2
                if len(info_cells) > player_info_idx:
                    cell = info_cells[player_info_idx]
                    divs = cell.find_all('div', recursive=False)
                    if len(divs) >= 3:
                        line1_txt = divs[0].get_text(strip=True)
                        id_match = re.search(r'(\d+)', line1_txt)
                        rank_span = divs[0].select_one("span")
                        entry['racer_id'] = int(id_match.group(1)) if id_match else None
                        entry['racer_rank'] = rank_span.text.strip() if rank_span else None
                        if entry['racer_rank'] is None:
                            if 'B2' in line1_txt: entry['racer_rank'] = 'B2'
                            elif 'B1' in line1_txt: entry['racer_rank'] = 'B1'
                            elif 'A2' in line1_txt: entry['racer_rank'] = 'A2'
                            elif 'A1' in line1_txt: entry['racer_rank'] = 'A1'
                        name_link = divs[1].select_one("a")
                        entry['racer_name'] = name_link.text.strip() if name_link else divs[1].get_text(strip=True)
                        detail_text = divs[2].get_text(separator=' ', strip=True)
                        pattern = re.compile(r'(?P<branch>[^/]+)\s*/\s*(?P<birthplace>[^/\s\d]+)\s*(?P<age>\d+)\s*歳\s*/?\s*(?P<weight>[\d\.]+)\s*kg', re.IGNORECASE)
                        match = pattern.search(detail_text)
                        if match:
                            entry['branch'] = match.group('branch').strip(); entry['birthplace'] = match.group('birthplace').strip()
                            entry['age'] = int(match.group('age')) if match.group('age') else None
                            entry['weight'] = float(match.group('weight')) if match.group('weight') else None
                        else: entry['branch'] = entry['birthplace'] = entry['age'] = entry['weight'] = None
                    else: main_logger.warning(f"  艇{boat_number}: 選手基本情報セル内のdiv数不足 ({len(divs)})")
                else: main_logger.warning(f"  艇{boat_number}: 選手基本情報セル(idx:{player_info_idx})不足")

                flst_idx = 3
                entry['f_count'] = 0; entry['l_count'] = 0; entry['avg_st'] = None
                if len(info_cells) > flst_idx:
                    flst_text_parts = info_cells[flst_idx].get_text(separator='\n', strip=True).split('\n')
                    for part in flst_text_parts:
                        if part.startswith('F'): entry['f_count'] = int(part[1:]) if part[1:].isdigit() else 0
                        elif part.startswith('L'): entry['l_count'] = int(part[1:]) if part[1:].isdigit() else 0
                        elif part != '-' and re.match(r'^[0-9\.]+$', part):
                            try: entry['avg_st'] = float(part)
                            except ValueError: pass
                else: main_logger.warning(f"  艇{boat_number}: F/L/STセル(idx:{flst_idx})不足")

                def safe_float_convert(text_list, list_idx):
                    val = None
                    if len(text_list) > list_idx and text_list[list_idx] != '-':
                        try: val = float(text_list[list_idx])
                        except (ValueError, TypeError): pass
                    return val

                nat_idx = 4
                entry['nat_win_rate'] = entry['nat_quinella_rate'] = entry['nat_trio_rate'] = None
                if len(info_cells) > nat_idx:
                    nat_text = info_cells[nat_idx].get_text(separator='\n', strip=True).split('\n')
                    entry['nat_win_rate'] = safe_float_convert(nat_text, 0)
                    entry['nat_quinella_rate'] = safe_float_convert(nat_text, 1)
                    entry['nat_trio_rate'] = safe_float_convert(nat_text, 2)
                else: main_logger.warning(f"  艇{boat_number}: 全国成績セル(idx:{nat_idx})不足")

                loc_idx = 5
                entry['loc_win_rate'] = entry['loc_quinella_rate'] = entry['loc_trio_rate'] = None
                if len(info_cells) > loc_idx:
                    loc_text = info_cells[loc_idx].get_text(separator='\n', strip=True).split('\n')
                    entry['loc_win_rate'] = safe_float_convert(loc_text, 0)
                    entry['loc_quinella_rate'] = safe_float_convert(loc_text, 1)
                    entry['loc_trio_rate'] = safe_float_convert(loc_text, 2)
                else: main_logger.warning(f"  艇{boat_number}: 当地成績セル(idx:{loc_idx})不足")

                motor_idx = 6
                entry['motor_no'] = entry['motor_quinella_rate'] = entry['motor_rate'] = None
                if len(info_cells) > motor_idx:
                    motor_text = info_cells[motor_idx].get_text(separator='\n', strip=True).split('\n')
                    entry['motor_no'] = int(motor_text[0]) if len(motor_text)>0 and motor_text[0].isdigit() else None
                    entry['motor_quinella_rate'] = safe_float_convert(motor_text, 1)
                    entry['motor_rate'] = entry['motor_quinella_rate']
                else: main_logger.warning(f"  艇{boat_number}: モーター情報セル(idx:{motor_idx})不足")

                boat_data_idx = 7
                entry['boat_no'] = entry['boat_quinella_rate'] = entry['boat_rate'] = None
                if len(info_cells) > boat_data_idx:
                    boat_text = info_cells[boat_data_idx].get_text(separator='\n', strip=True).split('\n')
                    entry['boat_no'] = int(boat_text[0]) if len(boat_text)>0 and boat_text[0].isdigit() else None
                    entry['boat_quinella_rate'] = safe_float_convert(boat_text, 1)
                    entry['boat_rate'] = entry['boat_quinella_rate']
                else: main_logger.warning(f"  艇{boat_number}: ボート情報セル(idx:{boat_data_idx})不足")

                # --- 今節成績 (prior_results) 解析 ---
                prior_results_str = ""
                try:
                    prior_res_row = tbody.select_one("tr.is-fBold")
                    if prior_res_row:
                        res_texts = [td.get_text(strip=True) for td in prior_res_row.select("td")]
                        cleaned_res = []
                        for t in res_texts:
                            if not t: continue
                            # 全角数字を半角に
                            t_norm = t.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
                            # 数字または特定の記号のみ採用
                            if t_norm.isdigit() or t_norm in ['F', 'L', 'S', 'K', '欠', '失', '転', '不']:
                                cleaned_res.append(t_norm)
                        prior_results_str = "".join(cleaned_res)
                except Exception as e_prior:
                    main_logger.warning(f"  艇{boat_number}: 今節成績解析エラー: {e_prior}")
                entry['prior_results'] = prior_results_str

                entries.append(entry)
                main_logger.debug(f"--- 艇 {boat_number} の解析正常終了 ---")

            except Exception as e:
                main_logger.error(f"!! 艇 {boat_number} の解析中に予期せぬエラー: {e}")
                main_logger.error(traceback.format_exc())
                main_logger.debug(f"--- 艇 {boat_number} の解析中断 ---")

        if not entries:
            main_logger.warning("  最終的に有効な出走情報を1件も取得できませんでした。")
            return None

        df = pd.DataFrame(entries)
        expected_cols = [
            'boat_number', 'racer_id', 'racer_rank', 'racer_name', 'branch', 'birthplace', 'age', 'weight',
            'f_count', 'l_count', 'avg_st',
            'nat_win_rate', 'nat_quinella_rate', 'nat_trio_rate',
            'loc_win_rate', 'loc_quinella_rate', 'loc_trio_rate',
            'motor_no', 'motor_quinella_rate', 'motor_rate',
            'boat_no', 'boat_quinella_rate', 'boat_rate'
        ]
        for col in expected_cols:
            if col not in df.columns:
                df[col] = None
                main_logger.debug(f"  DataFrameにカラム '{col}' が不足していたためNoneで作成しました。")

        main_logger.info(f"  出走表HTMLの解析完了。取得件数: {len(df)}")
        return df

    except Exception as e:
        main_logger.error(f"!! 出走表全体の解析プロセスでエラー: {e}")
        main_logger.error(traceback.format_exc())
        return None

def convert_direction_number_to_name(direction_num):
    """風向数値 (1-16) を方位名に変換"""
    if direction_num is None: return None
    directions = ["北", "北北東", "北東", "東北東", "東", "東南東", "南東", "南南東",
                 "南", "南南西", "南西", "西南西", "西", "西北西", "北西", "北北西"]
    try: index = int(direction_num) - 1; return directions[index] if 0 <= index < 16 else None
    except (ValueError, TypeError): return None

# --- 直前情報 解析 (旧スクリプト参照ロジック適用版) ---
def _parse_beforeinfo(html_content: str) -> dict | None:
    """直前情報HTML解析 (展示タイム、スタート展示、進入コース、気象情報)"""
    if not html_content:
        main_logger.warning("直前情報HTMLコンテンツが空です。")
        return None
    main_logger.debug("直前情報HTMLを解析中 (旧スクリプト参照ロジック適用)...")

    info = {
        'exhibition_times': {b: np.nan for b in range(1, 7)},
        'start_times': {b: np.nan for b in range(1, 7)},
        'exhibition_entry_courses': {b: None for b in range(1, 7)},
        'weather_text': None, 'wind_direction': None, 'wind_speed': None, 'wave_height': None,
        'wind_direction_name': None, 'temperature': None, 'water_temperature': None
    }
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        weather_el = soup.select_one("div.weather1_body")
        if weather_el:
            info['weather_text'] = weather_el.get_text(" ", strip=True)
            try:
                wind_dir_p = weather_el.select_one(".is-windDirection p.weather1_bodyUnitImage")
                if wind_dir_p and wind_dir_p.get('class'):
                    direction_class = next((cls for cls in wind_dir_p['class'] if cls.startswith('is-wind')), None)
                    if direction_class: info['wind_direction'] = int(re.sub(r'\D', '', direction_class)) if re.sub(r'\D', '', direction_class) else None
                wind_speed_span = weather_el.select_one(".is-wind span.weather1_bodyUnitLabelData")
                info['wind_speed'] = float(re.search(r'([\d\.]+)', wind_speed_span.text).group(1)) if wind_speed_span and re.search(r'([\d\.]+)', wind_speed_span.text) else None
                wave_height_span = weather_el.select_one(".is-wave span.weather1_bodyUnitLabelData")
                info['wave_height'] = float(re.search(r'([\d\.]+)', wave_height_span.text).group(1)) if wave_height_span and re.search(r'([\d\.]+)', wave_height_span.text) else None
                temp_title_span = weather_el.find("span", class_="weather1_bodyUnitLabelTitle", string="気温")
                if temp_title_span:
                    temp_data_span = temp_title_span.find_next_sibling("span", class_="weather1_bodyUnitLabelData")
                    if temp_data_span and re.search(r'([\d\.]+)', temp_data_span.text): info['temperature'] = float(re.search(r'([\d\.]+)', temp_data_span.text).group(1))
                water_temp_span = weather_el.select_one(".is-waterTemperature span.weather1_bodyUnitLabelData")
                if water_temp_span and re.search(r'([\d\.]+)', water_temp_span.text): info['water_temperature'] = float(re.search(r'([\d\.]+)', water_temp_span.text).group(1))
            except Exception as e_weather: main_logger.warning(f"    天候詳細数値の抽出中にエラー: {e_weather}")
        else: main_logger.warning("  天候情報(div.weather1_body)が見つかりません。")
        info['wind_direction_name'] = convert_direction_number_to_name(info.get('wind_direction'))

        main_table_exh_time = soup.select_one("table.is-w748")
        if main_table_exh_time:
            tbody_elements = main_table_exh_time.select("tbody.is-fs12")
            if len(tbody_elements) >= 6:
                for i, tbody in enumerate(tbody_elements[:6], 1):
                    boat_num = i; exh_time_cell = tbody.select_one("tr:first-child td:nth-of-type(5)")
                    if exh_time_cell:
                        exh_time_str = exh_time_cell.text.strip()
                        info['exhibition_times'][boat_num] = float(exh_time_str) if exh_time_str and exh_time_str != '--' else np.nan
                    else: main_logger.debug(f"  艇{boat_num}: 展示タイムセル(td:5)が見つかりません。")
            else: main_logger.warning(f"  展示タイムのtbody要素数が不足 ({len(tbody_elements)}個)。")
        else: main_logger.warning("  展示タイムのメインテーブル(table.is-w748)が見つかりません。")

        st_table = soup.select_one("table.is-w238")
        if st_table:
            rows = st_table.select("tbody tr")
            if len(rows) >= 6:
                for course_idx, row in enumerate(rows[:6]):
                    actual_course_number = course_idx + 1; boat_num_span = row.select_one("span.table1_boatImage1Number")
                    boat_num_int = int(boat_num_span.text.strip()) if boat_num_span and boat_num_span.text.strip().isdigit() else None
                    time_span = row.select_one("span.table1_boatImage1Time"); start_time_float = np.nan
                    if time_span:
                         st_time_str = time_span.text.strip()
                         if st_time_str.startswith("F"): start_time_float = float("0" + st_time_str[1:]) if len(st_time_str) > 1 and st_time_str[1:].replace('.','',1).isdigit() else np.nan
                         elif st_time_str and st_time_str != '--':
                              try: start_time_float = float(st_time_str)
                              except ValueError: pass
                    if boat_num_int is not None and 1 <= boat_num_int <= 6:
                        info['start_times'][boat_num_int] = start_time_float
                        info['exhibition_entry_courses'][boat_num_int] = actual_course_number
                        main_logger.debug(f"    STテーブル: 艇番 {boat_num_int}, コース {actual_course_number}, ST {start_time_float}")
                    else: main_logger.warning(f"  STテーブル コース{actual_course_number}行目: 有効な艇番({boat_num_span.text if boat_num_span else 'N/A'})が取得できませんでした。")
            else: main_logger.warning(f"  STテーブルの行数が不足 ({len(rows)}行)。")
        else: main_logger.warning("  STテーブル(table.is-w238)が見つかりません。")

        main_logger.debug(f"  _parse_beforeinfo 結果: exhibition_times={info['exhibition_times']}, exhibition_entry_courses={info['exhibition_entry_courses']}, start_times={info['start_times']}")

        if all(pd.isna(v) for v in info['exhibition_times'].values()) and \
           all(v is None for v in info['exhibition_entry_courses'].values()):
            main_logger.warning("  最終確認: 展示タイムおよび展示進入コースを全く取得できませんでした。")
        main_logger.debug(f"  直前情報HTMLの解析完了。進入コース: {info['exhibition_entry_courses']}")
        return info
    except Exception as e: main_logger.error(f"直前情報全体の解析エラー: {e}", exc_info=True); return None

# --- オッズ 解析 ---
def _get_safe_text(element) -> str: return element.get_text(strip=True) if element else ""
def _get_safe_float(text: str) -> float | None:
    try: return float(text.replace('倍', '').replace(',', ''))
    except (ValueError, TypeError): return None

# ★★★ オッズ解析修正 (ヘッダーから艇番のみ抽出) ★★★
def _parse_odds(html_content: str, logger: logging.Logger) -> dict | None:
    """3連単オッズ解析 (ヘッダー艇番抽出を修正)"""
    if not html_content: return None
    logger.info("オッズHTMLを解析中...")
    try:
        soup = BeautifulSoup(html_content, 'lxml')
        odds_data = {}
        target_tbody = soup.select_one("tbody.is-p3-0")
        if not target_tbody:
            target_tbody = soup.select_one("div.table1 > table > tbody")
            if not target_tbody: logger.error("エラー: オッズtbodyが見つかりません。"); return None

        first_places = []
        header_row = target_tbody.parent.select_one("thead tr")
        if header_row:
            th_tags = header_row.find_all('th', recursive=False)
            for th in th_tags:
                # is-boatColorクラスを持つthタグを探す
                if th.has_attr('class') and any('is-boatColor' in c for c in th['class']):
                    # thタグ内のテキスト全体から先頭の数字のみを抽出する
                    th_text = _get_safe_text(th)
                    num_match = re.match(r'^\s*(\d+)', th_text) # 先頭の数字を探す
                    if num_match:
                        num = num_match.group(1)
                        if num not in first_places: # 重複チェック
                            first_places.append(num)
                            if len(first_places) == 6: break # 6艇分見つかったら終了
        
        if len(first_places) != 6:
             logger.warning(f"オッズヘッダー解析: 1着艇ヘッダーが6つ検出できませんでした。デフォルト1-6使用。検出: {first_places}")
             first_places = [str(i) for i in range(1, 7)]

        rows = target_tbody.find_all('tr', recursive=False)
        num_cols = len(first_places)
        current_boat2 = [''] * num_cols; rowspan_remaining = [0] * num_cols

        for r_idx, row in enumerate(rows):
            cells = row.find_all('td', recursive=False); cell_ptr = 0
            for col_idx in range(num_cols):
                first_boat = first_places[col_idx]; boat2 = ""; boat3 = ""; odds_val = None
                try:
                    if rowspan_remaining[col_idx] > 0:
                        rowspan_remaining[col_idx] -= 1
                        if cell_ptr + 1 < len(cells):
                            boat2 = current_boat2[col_idx]
                            boat3 = _get_safe_text(cells[cell_ptr]); odds_val = _get_safe_float(_get_safe_text(cells[cell_ptr + 1]))
                            cell_ptr += 2
                        else: logger.warning(f"R{r_idx}C{col_idx+1} rowspan継続セル不足"); continue
                    else:
                        if cell_ptr >= len(cells): logger.warning(f"R{r_idx}C{col_idx+1} セル不足(ptr={cell_ptr})"); break
                        current_cell = cells[cell_ptr]
                        if current_cell.has_attr('rowspan'):
                            boat2_text = _get_safe_text(current_cell)
                            if boat2_text.isdigit():
                                current_boat2[col_idx] = boat2_text; boat2 = boat2_text
                                try: rowspan_remaining[col_idx] = max(0, int(current_cell['rowspan']) - 1)
                                except (ValueError, KeyError): rowspan_remaining[col_idx] = 0 # エラー時はrowspan=0扱い
                                if cell_ptr + 2 < len(cells):
                                    boat3 = _get_safe_text(cells[cell_ptr+1]); odds_val = _get_safe_float(_get_safe_text(cells[cell_ptr+2]))
                                    cell_ptr += 3
                                else: logger.warning(f"R{r_idx}C{col_idx+1} rowspan開始ペアセル不足"); cell_ptr += 1; continue
                            else: logger.warning(f"R{r_idx}C{col_idx+1} rowspanセル非数字'{boat2_text}'"); cell_ptr += 1; continue
                        else: logger.warning(f"R{r_idx}C{col_idx+1} rowspan期待箇所にrowspanなし"); cell_ptr +=1; continue
                    
                    if boat2.isdigit() and boat3.isdigit() and first_boat != boat2 and first_boat != boat3 and boat2 != boat3 and odds_val is not None and odds_val > 0:
                        odds_data[f"{first_boat}-{boat2}-{boat3}"] = odds_val
                except Exception as cell_err: logger.error(f"R{r_idx}C{col_idx+1}セル処理エラー: {cell_err}", exc_info=True)

        logger.info(f"  オッズ解析完了 (取得件数: {len(odds_data)})。")
        if len(odds_data) != 120: logger.warning(f"  ★警告★: 取得した3連単オッズが120通りではありません ({len(odds_data)}通り)。")
        return odds_data if odds_data else None
    except Exception as e: logger.error(f"オッズ全体の解析エラー: {e}", exc_info=True); return None


# --- データ取得統括関数 ---
async def get_realtime_data(date: str, place: str, race_no: str) -> dict | None:
    """指定レースのリアルタイムデータをまとめて取得・解析"""
    main_logger.info(f"--- {date} {place} {race_no}R リアルタイムデータ取得開始 ---")
    card_url = config.RACELIST_URL_FORMAT.format(rno=race_no, jcd=place, hd=date)
    before_url = config.BEFOREINFO_URL_FORMAT.format(rno=race_no, jcd=place, hd=date)
    odds_url = config.ODDS3T_URL_FORMAT.format(rno=race_no, jcd=place, hd=date)
    tasks = {'card_html': _fetch_url(card_url),
             'before_html': _fetch_url(before_url),
             'odds_html': _fetch_url(odds_url)}
    results = await asyncio.gather(*tasks.values())
    task_results = dict(zip(tasks.keys(), results))

    main_logger.info("  HTML解析処理中...")
    race_header_info = _parse_race_header_info(task_results.get('card_html'))
    racecard_df = _parse_racecard(task_results.get('card_html'))
    before_info = _parse_beforeinfo(task_results.get('before_html'))
    odds_info = _parse_odds(task_results.get('odds_html'), main_logger)

    if racecard_df is None or before_info is None or odds_info is None:
        main_logger.error("エラー: データ取得・解析に失敗。 (出走表 or 直前情報 or オッズ)")
        return None
    if all(pd.isna(v) for v in before_info['exhibition_times'].values()) and \
       all(v is None for v in before_info['exhibition_entry_courses'].values()):
        main_logger.warning("警告: 展示タイムおよび展示進入コースが全く取得できませんでした。")

    race_info_combined = {
        "date": date, "place": place, "race_no": int(race_no),
        "entries": racecard_df, "before_info": before_info,
        "race_title": race_header_info.get('race_title'),
        "day_num_str": race_header_info.get('day_num_str'),
        "race_grade_num": race_header_info.get('race_grade_num', 0),
        "is_night_race": race_header_info.get('is_night_race', 0),
        "tournament_day": race_header_info.get('tournament_day', 0),
    }
    main_logger.info(f"--- {date} {place} {race_no}R リアルタイムデータ取得完了 ---")
    return {"race_info": race_info_combined, "odds_info": odds_info}

# --- レース結果 解析 (修正版) ---
def _parse_all_race_results_from_html(html_content: bytes, logger: logging.Logger) -> dict:
    """払戻・結果HTMLから全てのレースの結果を解析する"""
    all_results = {}
    if not html_content:
        logger.warning("レース結果HTMLコンテンツが空です。")
        return all_results

    logger.info("レース結果HTMLを解析中 (全レース対象)...\n") # 改行を追加
    try:
        soup = BeautifulSoup(html_content, "lxml")

        # すべての div.table1 を取得a
        table1_divs = soup.select("div.table1")
        if not table1_divs:
            logger.warning("div.table1 要素が見つかりませんでした。")
            return all_results

        for div_idx, table1_div in enumerate(table1_divs):
            logger.debug(f"div.table1 ({div_idx + 1}/{len(table1_divs)}) を処理中...")
            main_table = table1_div.select_one("table") # div.table1 内の table を取得
            if not main_table:
                logger.warning(f"div.table1 ({div_idx + 1}) 内に table 要素が見つかりませんでした。")
                continue

            # ヘッダーからレース場名とコードを抽出
            place_codes_in_html = []
            header_row_places = main_table.select_one("thead tr:nth-of-type(1)")
            if header_row_places:
                for th in header_row_places.select("th[colspan=\"3\"]"):
                    img_tag = th.select_one("img[alt]")
                    if img_tag:
                        place_name = img_tag['alt']
                        # 正規化処理を追加
                        place_name = unicodedata.normalize('NFKC', place_name)
                        logger.debug(f"  HTMLから取得された場名 (img alt): {place_name!r}")
                        # config.PLACE_CODE_TO_NAME を逆引きしてコードを取得
                        try:
                            from . import config
                        except ImportError:
                            import config # スクリプトとして直接実行する場合のフォールバック

                        found_code = next((code for code, name in config.PLACE_CODE_TO_NAME.items() if name == place_name), None)

                        if found_code:
                            place_codes_in_html.append(found_code)
                            logger.debug(f"  HTMLから抽出された場名: {place_name!r}, 対応するコード: {found_code!r}")
                        else:
                            logger.warning(f"  不明なレース場名がヘッダーにありました: {place_name!r} (configに存在しないか、表記揺れ)。")
                    else:
                        logger.warning("レース場名画像が見つかりません。")

            if not place_codes_in_html:
                logger.error(f"div.table1 ({div_idx + 1}) のHTMLヘッダーから有効なレース場コードを抽出できませんでした。")
                continue

            logger.debug(f"HTMLから検出されたレース場コード: {place_codes_in_html}")

            # 各レースのtbodyをループ
            race_tbodies = main_table.select("tbody")

            for tbody in race_tbodies:
                race_num_th = tbody.select_one("th.is-thColor8")
                if not race_num_th:
                    logger.debug("レース番号のthが見つからないtbodyをスキップします。")
                    continue

                current_race_no = race_num_th.get_text(strip=True).replace("R", "")

                # 各レース場の結果データ (<td>) を3つずつのブロックに分けて処理
                td_cells = tbody.select("td")

                # 3つずつのブロックに分割
                for i in range(0, len(td_cells), 3):
                    combo_td = td_cells[i]
                    payout_td = td_cells[i+1]
                    popularity_td = td_cells[i+2]

                    # このブロックがどのレース場のものか特定
                    # place_codes_in_html のインデックスと対応
                    current_place_idx = i // 3
                    if current_place_idx >= len(place_codes_in_html):
                        logger.warning(f"レース場コードのインデックスが範囲外です。i={i}, current_place_idx={current_place_idx}, len(place_codes_in_html)={len(place_codes_in_html)}")
                        continue

                    current_place_code = place_codes_in_html[current_place_idx]

                    results = {}

                    # --- 組番 (Winning Combo) ---
                    sanrentan_combo_div = combo_td.select_one("div.numberSet1_row")
                    if sanrentan_combo_div:
                        sanrentan_combo_spans = sanrentan_combo_div.select("span.numberSet1_number")
                        if len(sanrentan_combo_spans) == 3:
                            sanrentan_combination = "-".join([s.get_text(strip=True) for s in sanrentan_combo_spans])
                            results["sanrentan_combination"] = sanrentan_combination
                            results["status"] = "finished"
                        else:
                            logger.warning(f"  {current_place_code}場 {current_race_no}R: 組番のspanタグが3つではありませんでした。")
                            continue # このレースの結果は不完全なのでスキップ
                    else:
                        # 組番divがない場合、結果はまだ出ていないか、中止
                        # 締切時刻表示のリンクがあるか確認
                        deadline_link = tbody.select_one(f"a[href*=\'racelist\'][href*=\'jcd={current_place_code}\'][href*=\'rno={current_race_no}\']")
                        if deadline_link:
                            logger.info(f"{current_place_code}場 {current_race_no}R はまだ終了していません（締切時刻表示）。")
                            continue # まだ結果が出ていないのでスキップ
                        elif "中止" in combo_td.get_text() or "返" in combo_td.get_text(): # 中止または返還のテキストがあるか
                            logger.warning(f"{current_place_code}場 {current_race_no}R は中止または返還と判断しました。")
                            results["status"] = "canceled"
                        else:
                            logger.warning(f"{current_place_code}場 {current_race_no}R は結果も締切時刻も中止情報も見つかりませんでした。HTML構造変更の可能性。")
                            continue # 結果が不明なのでスキップ

                    # --- 3連単 (Payout) ---
                    payout_span = payout_td.select_one("span.is-payout2")
                    sanrentan_payout = None
                    if payout_span:
                        payout_str = payout_span.get_text(strip=True)
                        payout_val_match = re.search(r"[\d,]+", payout_str)
                        if payout_val_match:
                            sanrentan_payout = int(payout_val_match.group(0).replace(",", ""))
                    results["sanrentan_payout"] = sanrentan_payout

                    # --- 人気 (Popularity) ---
                    # td > a または td > span.is-fBold を考慮
                    popularity_el = popularity_td.select_one("a") or popularity_td.select_one("span.is-fBold")
                    popularity = None
                    if popularity_el:
                        popularity_text = popularity_el.get_text(strip=True)
                        if popularity_text.isdigit():
                            popularity = int(popularity_text)
                        elif popularity_text == "返":
                            popularity = "返" # 返還の場合
                    results["popularity"] = popularity

                    all_results[(current_place_code, current_race_no)] = results

        logger.info(f"レース結果の解析完了。取得件数: {len(all_results)}")
        return all_results

    except Exception as e:
        logger.error(f"レース結果の解析中にエラーが発生: {e}", exc_info=True)
        return all_results # エラーが発生しても、これまでに取得した結果は返す

def _parse_race_results(html_content: bytes, place_code: str, race_no: str, logger: logging.Logger) -> dict | None:
    """
    指定されたHTMLコンテンツから、特定の場コードとレース番号のレース結果を解析して返す。
    """
    all_results = _parse_all_race_results_from_html(html_content, logger)
    target_key = (place_code, race_no)
    return all_results.get(target_key)

# --- レース結果取得統括関数 ---
async def get_race_result(date: str, place_code: str, race_no: str) -> dict | None:
    """指定レースのレース結果をまとめて取得・解析"""
    main_logger.info(f"--- {date} {place_code} {race_no}R レース結果取得開始 ---")
    # payエンドポイントは日付のみを使用
    result_url = config.RESULT_URL_FORMAT.format(hd=date)

    result_html = await _fetch_url(result_url)
    if result_html is None:
        main_logger.error(f"エラー: レース結果HTMLの取得に失敗しました。URL: {result_url}")
        return None

    # _parse_all_race_results_from_html はその日の全レース結果を返す
    all_results_for_day = _parse_all_race_results_from_html(result_html, main_logger)

    # 必要なレースの結果を抽出
    target_key = (place_code, race_no)
    if target_key in all_results_for_day:
        main_logger.info(f"--- {date} {place_code} {race_no}R レース結果取得完了 ---")
        return all_results_for_day[target_key]
    else:
        main_logger.warning(f"--- {date} {place_code} {race_no}R のレース結果が見つかりませんでした。 ---")
        return None