"""
find_active_momentum.py
現在時刻で展示情報が発表されているナイターレースから、開催2日目以降（モメンタム計算可能）の未確定レースを高速自動探索・ダンプするスクリプト
"""

import os
import sys
import datetime
import sqlite3
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# プロジェクトルートのパス追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_boatrace import BoatRaceScraper, VENUE_MAP
from auto_trader import FeatureEngineer, fetch_series_momentum


# 対象ナイター会場
NIGHTER_VENUES = [
    (1, '桐生'),
    (7, '蒲郡'),
    (12, '住之江'),
    (15, '丸亀'),
    (19, '下関'),
    (20, '若松'),
    (24, '大村'),
]

def find_active_momentum_race():
    today_str = datetime.date.today().strftime('%Y%m%d')
    formatted_date = f"{today_str[:4]}-{today_str[4:6]}-{today_str[6:]}"
    
    print("=========================================================================")
    print(f" 🔍 [LIVE SEARCH] 現在時刻でモメンタム取得可能な未確定レースの高速探索 ({formatted_date})")
    print("=========================================================================")

    # 選手名の静的マッピングをロード
    name_map = {}
    try:
        r_params = pd.read_csv('app_data/static_racer_params.csv')
        name_map = dict(zip(r_params['racer_id'].astype(int), r_params['racer_name']))
    except Exception:
        pass

    db_path = 'boatrace.db'

    # 現在時刻に合わせて夕方〜夜の直近レース順（R5, R6, R7, R8, R4, R9, R10, R11, R12, R3, R2, R1）でスマート探索
    smart_race_order = [6, 7, 5, 8, 4, 9, 10, 11, 12, 3, 2, 1]

    for venue_code, venue_name in NIGHTER_VENUES:
        print(f"\n🏟️ 会場チェック: {venue_name} ({venue_code:02d}場)...")
        
        for r_no in smart_race_order:
            print(f"   ▶ {venue_name} 第{r_no}レース 確認中...", end=" ", flush=True)
            
            try:
                df_scraped = BoatRaceScraper.get_race_data(today_str, venue_code, r_no)
            except Exception as e:
                print(f"エラースキップ: {e}")
                continue

            if df_scraped is None or df_scraped.empty:
                print("❌ 直前情報なし (スキップ)")
                continue

            if 'exhibition_time' not in df_scraped.columns:
                print("❌ 展示タイム未発表 (スキップ)")
                continue

            valid_ex = df_scraped['exhibition_time'].dropna().tolist()
            if len(valid_ex) < 6 or any(t <= 0 for t in valid_ex):
                print(f"❌ 展示タイム不完全 ({len(valid_ex)}/6)")
                continue

            # レース結果の確認（未確定レースか確認）
            res = BoatRaceScraper.get_race_result(today_str, venue_code, r_no)
            is_unconfirmed = (res is None)

            status_tag = "🔥 直前情報発表済み（未確定/締切前）" if is_unconfirmed else "🏁 結果確定済み"
            print(f"✅ 展示タイム取得！ [{', '.join([f'{t:.2f}' for t in valid_ex])}] ({status_tag})")

            # 選手名の補完
            df_scraped['racer_name'] = df_scraped['racer_id'].astype(int).map(name_map).fillna('選手')

            # 特徴量エンジンの実行
            df_feat = FeatureEngineer.process(df_scraped.copy(), venue_name=venue_name, race_date=formatted_date)

            # 6艇のうち 0.0 以外のモメンタム差分を持つ選手がいるか判定
            mom_diffs = df_feat['ex_momentum_diff'].tolist()
            non_zero_count = sum(1 for d in mom_diffs if abs(d) > 1e-5)

            if non_zero_count > 0:
                print(f"\n{'🎯'*25}")
                print(f"🎉 モメンタム計算可能なアクティブレースを発見！ ({venue_name} 第{r_no}レース - {status_tag})")
                print(f"{'🎯'*25}\n")

                # DBから節間履歴を取得して詳細表示用データを準備
                racer_ids = df_scraped['racer_id'].astype(int).tolist()
                ph = ','.join(['?'] * len(racer_ids))
                conn = sqlite3.connect(db_path)
                query_history = f"""
                SELECT r.race_id, r.race_date, r.race_number, re.racer_id, bi.exhibition_time
                FROM before_info bi
                JOIN races r ON bi.race_id = r.race_id
                JOIN race_entries re ON bi.race_id = re.race_id AND bi.boat_number = re.boat_number
                WHERE CAST(r.venue_code AS INTEGER) = ?
                  AND re.racer_id IN ({ph})
                  AND r.race_date >= date(?, '-7 days')
                  AND r.race_date <= ?
                  AND bi.exhibition_time > 0
                ORDER BY re.racer_id, r.race_date ASC, r.race_number ASC
                """
                df_history = pd.read_sql_query(query_history, conn, params=[venue_code] + racer_ids + [formatted_date, formatted_date])
                conn.close()

                # 結果のダンプ出力
                print(f"📍 レース情報: {formatted_date} | {venue_name} ({venue_code:02d}場) | 第{r_no}レース [{status_tag}]")
                print(f"🌤️ 気象情報: 風速 {df_scraped['wind_speed'].iloc[0]}m, 波高 {df_scraped['wave_height'].iloc[0]}cm")
                print("\n📊 【モメンタム抽出結果一覧】")
                print("-" * 125)
                print(f"{'艇':<2} | {'選手名':<10} | {'登番':<5} | {'当日展示':<8} | {'前走(DB)':<8} | {'節間平均':<6} | {'節間走数':<4} | {'ex_diff_min':<11} | {'ex_momentum_diff':<18} | {'ex_momentum_dev':<15}")
                print("-" * 125)

                for idx, row in df_feat.iterrows():
                    b_num = int(row['boat_number'])
                    r_id = int(row['racer_id'])
                    r_name = str(row['racer_name'])
                    cur_ex = float(row['exhibition_time'])
                    
                    r_hist = df_history[df_history['racer_id'] == r_id]
                    past_times = r_hist['exhibition_time'].tolist()
                    prev_ex_str = f"{past_times[-1]:.2f}" if len(past_times) >= 1 else "---"
                    
                    combined_times = past_times + [cur_ex] if cur_ex > 0 else past_times
                    mean_ex_str = f"{np.mean(combined_times):.2f}" if len(combined_times) >= 1 else "---"
                    total_runs = len(combined_times)

                    ex_diff_min = float(row['ex_diff_from_race_min'])
                    ex_mom_diff = float(row['ex_momentum_diff'])
                    ex_mom_dev = float(row['ex_momentum_deviation'])

                    if abs(ex_mom_diff) > 1e-5:
                        signal_icon = "🚀 (気配良化)" if ex_mom_diff < 0 else "⚠️ (気配低下)"
                    else:
                        signal_icon = "➖ (変化なし/初走)"

                    print(f"{b_num:<2} | {r_name:<10} | {r_id:<5} | {cur_ex:<8.2f} | {prev_ex_str:<8} | {mean_ex_str:<8} | {total_runs:<6} | {ex_diff_min:<11.3f} | {ex_mom_diff:<7.3f} {signal_icon:<14} | {ex_mom_dev:<15.3f}")

                print("-" * 125)
                print(f"\n✅ 診断: 6艇中 {non_zero_count} 艇で有効な節間モメンタムをリアルタイム算出しました。自動検索を終了します。\n")
                return True

    print("\n⚠️ 現在展示情報が発表されているナイターレースで、モメンタム計算可能な対象は見つかりませんでした。")
    return False

if __name__ == '__main__':
    find_active_momentum_race()
