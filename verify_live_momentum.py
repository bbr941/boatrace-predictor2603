"""
verify_live_momentum.py
本番推論パイプラインにおける節間代替モメンタム特徴量のデバッグ・検証スクリプト
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np

# プロジェクトルートのパス追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from auto_trader import FeatureEngineer, fetch_series_momentum

def get_venue_name(venue_code: int) -> str:
    venue_names = {
        1: '桐生', 2: '戸田', 3: '江戸川', 4: '平和島', 5: '多摩川',
        6: '浜名湖', 7: '蒲郡', 8: '常滑', 9: '津', 10: '三国',
        11: 'びわこ', 12: '住之江', 13: '尼崎', 14: '鳴門', 15: '丸亀',
        16: '児島', 17: '宮島', 18: '徳山', 19: '下関', 20: '若松',
        21: '芦屋', 22: '福岡', 23: '唐津', 24: '大村'
    }
    return venue_names.get(int(venue_code), '住之江')

def verify_race_momentum(race_id: str, db_path: str = 'boatrace.db'):
    print(f"\n{'='*80}")
    print(f"🔍 [検証対象レースID: {race_id}]")
    print(f"{'='*80}")

    conn = sqlite3.connect(db_path)
    
    # 1. レース基本情報と環境情報の取得
    query_race = f"""
    SELECT r.race_id, r.race_date, r.venue_code, r.race_number,
           r.wind_speed, r.wind_direction, r.wave_height
    FROM races r
    WHERE r.race_id = '{race_id}'
    """
    df_race_info = pd.read_sql_query(query_race, conn)
    if df_race_info.empty:
        print(f"❌ レースID {race_id} が見つかりませんでした。")
        conn.close()
        return

    race_row = df_race_info.iloc[0]
    race_date = str(race_row['race_date'])
    venue_code = int(race_row['venue_code'])
    venue_name = get_venue_name(venue_code)
    race_num = int(race_row['race_number'])

    print(f"📍 開催情報: {race_date} | {venue_name} (場コード: {venue_code:02d}) | 第{race_num}レース")
    print(f"🌤️ 気象情報: 風速 {race_row['wind_speed']}m, 風向 {race_row['wind_direction']}, 波高 {race_row['wave_height']}cm")

    # 2. 出走表と直前情報の取得
    query_entries = f"""
    SELECT re.race_id, re.boat_number, re.racer_id, re.racer_name, re.racer_rank,
           re.motor_rate, re.boat_rate, re.weight, re.branch, re.nat_win_rate,
           re.nat_quinella_rate, re.loc_win_rate, re.loc_quinella_rate, re.prior_results,
           bi.exhibition_time, bi.exhibition_start_timing
    FROM race_entries re
    LEFT JOIN before_info bi ON re.race_id = bi.race_id AND re.boat_number = bi.boat_number
    WHERE re.race_id = '{race_id}'
    ORDER BY re.boat_number
    """
    df_entries = pd.read_sql_query(query_entries, conn)
    
    # 環境カラムの付与
    df_entries['wind_speed'] = race_row['wind_speed']
    df_entries['wind_direction'] = race_row['wind_direction']
    df_entries['wave_height'] = race_row['wave_height']
    df_entries['venue_code'] = venue_code
    df_entries['pred_course'] = df_entries['boat_number']

    # 3. 節間の生履歴（展示タイム）を直接クエリして検証用データを抽出
    racer_ids = df_entries['racer_id'].tolist()
    ph = ','.join(['?'] * len(racer_ids))
    query_history = f"""
    SELECT r.race_id, r.race_date, r.race_number, re.boat_number, re.racer_id, re.racer_name, bi.exhibition_time
    FROM before_info bi
    JOIN races r ON bi.race_id = r.race_id
    JOIN race_entries re ON bi.race_id = re.race_id AND bi.boat_number = re.boat_number
    WHERE CAST(r.venue_code AS INTEGER) = ?
      AND re.racer_id IN ({ph})
      AND r.race_date <= ?
      AND bi.exhibition_time > 0
    ORDER BY re.racer_id, r.race_date ASC, r.race_number ASC
    """
    df_history = pd.read_sql_query(query_history, conn, params=[venue_code] + racer_ids + [race_date])
    conn.close()

    # 4. FeatureEngineer による本番パイプライン特徴量生成
    print("\n⚙️ FeatureEngineer.process を実行して本番特徴量を生成中...")
    df_feat = FeatureEngineer.process(df_entries.copy(), venue_name=venue_name, race_date=race_date)

    # 5. モメンタム特徴量の抽出とダンプ
    print("\n📊 【モメンタム & 展示気配 特徴量ダンプ】")
    print("-" * 125)
    print(f"{'艇':<2} | {'選手名':<8} | {'登番':<5} | {'当日展示':<6} | {'前走展示':<6} | {'節間平均':<6} | {'節間走数':<4} | {'ex_diff_min':<11} | {'ex_momentum_diff':<18} | {'ex_momentum_dev':<15}")
    print("-" * 125)

    non_zero_momentum_count = 0

    for idx, row in df_feat.iterrows():
        b_num = int(row['boat_number'])
        r_id = int(row['racer_id'])
        r_name = str(row['racer_name'])
        cur_ex = float(row['exhibition_time'])
        
        # 履歴から前走・平均を計算
        r_hist = df_history[df_history['racer_id'] == r_id]
        times = r_hist['exhibition_time'].tolist()
        num_runs = len(times)
        prev_ex_str = f"{times[-2]:.2f}" if num_runs >= 2 else "---"
        mean_ex_str = f"{np.mean(times):.2f}" if num_runs >= 1 else "---"
        
        ex_diff_min = float(row['ex_diff_from_race_min'])
        ex_mom_diff = float(row['ex_momentum_diff'])
        ex_mom_dev = float(row['ex_momentum_deviation'])

        if abs(ex_mom_diff) > 1e-5:
            non_zero_momentum_count += 1
            signal_icon = "🚀 (気配良化)" if ex_mom_diff < 0 else "⚠️ (気配低下)"
        else:
            signal_icon = "➖ (変化なし/初走)"

        print(f"{b_num:<2} | {r_name:<8} | {r_id:<5} | {cur_ex:<8.2f} | {prev_ex_str:<8} | {mean_ex_str:<8} | {num_runs:<6} | {ex_diff_min:<11.3f} | {ex_mom_diff:<7.3f} {signal_icon:<14} | {ex_mom_dev:<15.3f}")

    print("-" * 125)

    # 6. 異常検知・判定
    print("\n🔍 【異常検知・整合性診断】")
    if non_zero_momentum_count > 0:
        print(f"✅ 判定: 正常稼働！ 6艇中 {non_zero_momentum_count} 艇で有効な節間モメンタムシグナル（±の有意な差分）を検出しました。")
        print("   -> 過去展示タイムとのタイム差 (ex_momentum_diff) および 節間平均偏差 (ex_momentum_deviation) が本番パイプラインで正しく計算されています。")
    else:
        max_runs = max([len(df_history[df_history['racer_id'] == rid]) for rid in racer_ids]) if not df_history.empty else 0
        if max_runs <= 1:
            print("ℹ️ 判定: 節間初走/節初日。今節の過去走が存在しないため、安全なデフォルト値 (0.0) が適用されています（仕様通り）。")
        else:
            print("⚠️ 判定: 異常検知！ 過去データが存在するにもかかわらず ex_momentum_diff が 0.0 になっています。クエリまたはデータフローを確認してください。")

def main():
    print("=========================================================================")
    print(" 🚀 BOATRACE REALTIME PIPELINE: LIVE MOMENTUM VERIFICATION & DEBUG ")
    print("=========================================================================")
    
    test_races = [
        ('01_20260828_12', "桐生 (01場) 2026-08-28 12R [節4日目 特選/優勝戦]"),
        ('07_20260828_12', "蒲郡 (07場) 2026-08-28 12R [節6日目 優勝戦]"),
        ('13_20260828_12', "尼崎 (13場) 2026-08-28 12R [節6日目 優勝戦]"),
    ]

    for race_id, desc in test_races:
        print(f"\n▶ 対象: {desc}")
        verify_race_momentum(race_id)

if __name__ == '__main__':
    main()
