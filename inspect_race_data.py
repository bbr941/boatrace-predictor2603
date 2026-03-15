
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import sys

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

class BoatRaceScraper:
    @staticmethod
    def get_soup(url):
        print(f"Fetching {url}")
        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=15)
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding
                return BeautifulSoup(resp.text, 'html.parser')
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(1)
        return None

    @staticmethod
    def parse_float(text):
        try:
            return float(re.search(r'([\d\.]+)', text).group(1))
        except:
            return 0.0

    @staticmethod
    def get_odds(date_str, venue_code, race_no):
        jcd = f"{int(venue_code):02d}"
        url = f"https://www.boatrace.jp/owpc/pc/race/oddstf?rno={race_no}&jcd={jcd}&hd={date_str}"
        soup = BoatRaceScraper.get_soup(url)
        odds_map = {}
        if soup:
            try:
                tables = soup.select("table.is-w495")
                target_table = None
                for t in tables:
                     if "単勝" in t.get_text():
                         target_table = t
                         break
                
                if target_table:
                     rows = target_table.select("tbody tr")
                     for row in rows:
                         tds = row.select("td")
                         if len(tds) >= 3:
                             try:
                                 bn_txt = tds[0].get_text(strip=True)
                                 bn = int(bn_txt)
                                 val_txt = tds[2].get_text(strip=True)
                                 val = float(val_txt)
                                 if val > 0:
                                     odds_map[bn] = 1.0 / val
                             except: pass
            except: pass
        return odds_map

    @staticmethod
    def get_race_data(date_str, venue_code, race_no):
        jcd = f"{int(venue_code):02d}"
        url_before = f"https://www.boatrace.jp/owpc/pc/race/beforeinfo?rno={race_no}&jcd={jcd}&hd={date_str}"
        url_list = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={race_no}&jcd={jcd}&hd={date_str}"
        
        soup_before = BoatRaceScraper.get_soup(url_before)
        soup_list = BoatRaceScraper.get_soup(url_list)
        
        odds_map = BoatRaceScraper.get_odds(date_str, venue_code, race_no)
        
        if not soup_before or not soup_list:
            print("Failed to get soup.")
            return None
            
        weather = {'wind_direction': 0, 'wind_speed': 0.0, 'wave_height': 0.0}
        try:
            w = soup_before.select_one("div.weather1_body")
            if w:
                ws = w.select_one(".is-wind span.weather1_bodyUnitLabelData")
                if ws: weather['wind_speed'] = BoatRaceScraper.parse_float(ws.text)
                wh = w.select_one(".is-wave span.weather1_bodyUnitLabelData")
                if wh: weather['wave_height'] = BoatRaceScraper.parse_float(wh.text)
                print(f"Weather: {weather}")
        except: pass
        
        boat_before = {}
        try:
            # Inspection of Before Info Table (Exhibition Time)
            print("\n--- Inspecting Before Info (Exhibition Time) ---")
            # Usually one table per boat? Or one big table?
            # app_boatrace.py assumes multiple tbodies in .is-w748
            tables_before = soup_before.select("table.is-w748 tbody")
            print(f"Found {len(tables_before)} Before-Info blocks.")
            
            for i, tb in enumerate(tables_before):
                tds = tb.select("td")
                print(f"Block {i+1}: Found {len(tds)} columns.")
                # Debug all columns
                col_texts = [td.get_text(strip=True) for td in tds]
                print(f"  Cols: {col_texts}")
                
                ex_val = 0.0
                has_ex_time = False
                
                if len(tds) >= 5:
                    ex_txt = tds[4].get_text(strip=True)
                    print(f"  ExTime Raw: '{ex_txt}'")
                    if ex_txt and ex_txt != '\xa0': # Check for empty or nbsp
                        try:
                            ex_val = float(ex_txt)
                            has_ex_time = True
                        except: pass
                
                boat_before[i+1] = {
                    'ex_time': ex_val if has_ex_time else None, # key differentiator?
                    'has_ex_time': has_ex_time
                }
            
            # Inspection of ST
            print("\n--- Inspecting Before Info (Start Timing) ---")
            for idx, row in enumerate(soup_before.select("table.is-w238 tbody tr")):
                bn_span = row.select_one("span.table1_boatImage1Number")
                if bn_span:
                    b = int(bn_span.text.strip())
                    st_span = row.select_one("span.table1_boatImage1Time")
                    val = 0.20
                    raw_txt = ""
                    if st_span:
                        raw_txt = st_span.text.strip()
                        if 'L' in raw_txt: val = 1.0
                        elif 'F' in raw_txt: val = -0.05
                        else: val = BoatRaceScraper.parse_float(raw_txt)
                    
                    if b not in boat_before: boat_before[b] = {}
                    boat_before[b]['st'] = val
                    boat_before[b]['pred_course'] = idx + 1
        except Exception as e:
            print(f"Before Info Error: {e}")

        rows = []
        try:
            print("\n--- Inspecting Race List (Main Data) ---")
            tbodies = soup_list.select("tbody.is-fs12")
            
            for i, tb in enumerate(tbodies):
                bn = i + 1
                tds = tb.select("td")
                
                # Print all column texts for the first boat to debug indices
                if i == 0:
                    print(f"Boat 1 Columns ([Index] Content):")
                    for idx, td in enumerate(tds):
                        print(f"  [{idx}] {td.get_text(' ', strip=True)}")

                # Motor/Boat parsing
                motor = 0.0
                boat = 0.0
                
                # Default indices in app_boatrace.py: Motor=6, Boat=7
                if len(tds) > 7:
                    m_txt = tds[6].get_text(" ", strip=True)
                    b_txt = tds[7].get_text(" ", strip=True)
                    
                    # Parse Motor
                    parts_m = m_txt.split()
                    if len(parts_m) >= 2: motor = float(parts_m[1].replace('%', '')) # "20 30.5" -> 30.5
                    
                    # Parse Boat
                    parts_b = b_txt.split()
                    if len(parts_b) >= 2: boat = float(parts_b[1].replace('%', ''))
                
                is_absent = False
                # Check absence based on Ex Time
                ex_info = boat_before.get(bn, {})
                ex_time_val = ex_info.get('ex_time')
                has_ex = ex_info.get('has_ex_time', False)
                
                if not has_ex:
                    is_absent = True
                    print(f"Boat {bn}: Marked ABSENT (No Exhibition Time)")
                else:
                    print(f"Boat {bn}: Present (Ex Time {ex_time_val})")

                row = {
                    'race_id': f"{date_str}_{venue_code}_{race_no}",
                    'boat_number': bn,
                    'motor_rate': motor,
                    'boat_rate': boat,
                    'exhibition_time': ex_time_val,
                    'is_absent': is_absent
                }
                rows.append(row)
                
        except Exception as e:
            print(f"Race List Error: {e}")
            return None
            
        return pd.DataFrame(rows)

if __name__ == "__main__":
    # Kiryu (01), 2025-12-27, 12R
    df = BoatRaceScraper.get_race_data("20251227", "01", 12)
    if df is not None:
        print("\n=== Final Extracted DataFrame ===")
        print(df)
    else:
        print("No Data Returned.")
