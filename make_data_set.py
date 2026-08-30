import os
import sqlite3
import pandas as pd
import numpy as np
import argparse
import re

# データベースパス (ローカルDBの存在確認とフォールバック)
LOCAL_DB_CANDIDATES = [
    r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db',
    'boatrace.db'
]

def get_db_path():
    for p in LOCAL_DB_CANDIDATES:
        if os.path.exists(p):
            return p
    return 'boatrace.db'

DB_PATH = get_db_path()

def get_connection():
    norm_path = os.path.abspath(DB_PATH).replace('\\', '/')
    try:
        return sqlite3.connect(f"file:///{norm_path}?mode=ro", uri=True)
    except Exception:
        return sqlite3.connect(DB_PATH)

def load_base_data(conn, limit=None, start_date=None):
    """
    基本データと統計データを結合して取得する
    limit: テスト用に取得件数を制限する場合に指定
    start_date: 開始日付 (例: '2024-01-01')
    """
    where_clause = ""
    if start_date:
        where_clause = f"WHERE r.race_date >= '{start_date}' AND r.is_cancelled = 0"
    else:
        where_clause = "WHERE r.is_cancelled = 0"

    query = f"""
    SELECT
        -- 識別子
        re.race_id,
        re.boat_number,
        re.racer_id,
        r.race_date,
        r.venue_code,
        
        -- 直前情報 & 進入予想
        bi.exhibition_time,
        bi.exhibition_start_timing,
        COALESCE(bi.exhibition_entry_course, re.boat_number) as pred_course,
        
        -- 選手能力 (絶対評価)
        re.nat_win_rate,
        re.nat_quinella_rate,
        re.motor_rate,
        re.boat_rate,
        re.prior_results,
        re.weight,
        re.branch,
        re.racer_rank,
        
        -- レース結果 (rank) from results
        res.finish_order as rank,
        
        -- コース別成績 (Racer_CourseStats)
        rcs.RacesRun as course_run_count,
        rcs.QuinellaRate as course_quinella_rate,
        rcs.TrifectaRate as course_trifecta_rate,
        rcs.FirstPlaceRate as course_1st_rate,
        rcs.AvgStartTiming as course_avg_st,
        
        -- 決まり手実績 (Racer_CourseWinTech)
        COALESCE(wt.RacesRun, 0) as wintech_races_run,
        COALESCE(wt.Wins, 0) as wintech_wins,
        COALESCE(wt.Nige, 0) as nige_count,
        COALESCE(wt.Makuri, 0) as makuri_count,
        COALESCE(wt.Makurizashi, 0) as makurizashi_count,
        COALESCE(wt.Sashi, 0) as sashi_count,
        
        -- 会場・環境情報
        r.wind_speed,
        r.wind_direction,
        r.wave_height,
        r.weather,
        v.venue_name,
        
        -- 会場別成績 (Racer_VenueStats)
        rvs.WinRate as local_win_rate,
        
        -- 会場・コース別傾向 (course_rates)
        cr.rate_1st as venue_course_1st_rate,
        cr.rate_2nd as venue_course_2nd_rate,
        cr.rate_3rd as venue_course_3rd_rate

    FROM race_entries re
    JOIN races r ON re.race_id = r.race_id
    JOIN venues v ON r.venue_code = v.venue_code
    LEFT JOIN before_info bi ON re.race_id = bi.race_id AND re.boat_number = bi.boat_number
    
    -- 結果の結合 (for Label)
    LEFT JOIN results res ON re.race_id = res.race_id AND re.boat_number = res.boat_number
    
    -- コース別成績の結合 (展示進入コース予測を使用)
    LEFT JOIN Racer_CourseStats rcs 
        ON re.racer_id = rcs.RacerID 
        AND rcs.Course = COALESCE(bi.exhibition_entry_course, re.boat_number)
        
    -- 決まり手の結合 (まくり差し等を含む)
    LEFT JOIN Racer_CourseWinTech wt 
        ON re.racer_id = wt.RacerID 
        AND wt.Course = COALESCE(bi.exhibition_entry_course, re.boat_number)
        
    -- 会場別成績の結合
    LEFT JOIN Racer_VenueStats rvs
        ON re.racer_id = rvs.RacerID
        AND v.venue_name = rvs.Venue
        
    -- 会場コース傾向の結合
    LEFT JOIN course_rates cr
        ON r.venue_code = cr.venue_code
        AND cr.course_number = COALESCE(bi.exhibition_entry_course, re.boat_number)
    
    {where_clause}
    """
    
    if limit:
        query += f" LIMIT {limit}"
        
    df = pd.read_sql(query, conn)
    if not df.empty and 'race_date' in df.columns:
        df = df.sort_values(['race_date', 'race_number', 'boat_number'], ascending=[False, False, True]).reset_index(drop=True)
    return df



def load_st_stability(conn, limit=None):
    """
    resultsテーブルから直近のST標準偏差を計算する
    """
    print("Calculating ST Standard Deviation from results...")
    query = "SELECT racer_id, start_timing FROM results WHERE start_timing IS NOT NULL"
    if limit:
        query += f" LIMIT {limit}"
    df_res = pd.read_sql(query, conn)
    
    # 欠損除去
    df_res = df_res.dropna()
    
    # 計算 (選手ごとのST標準偏差)
    st_stats = df_res.groupby('racer_id')['start_timing'].std().reset_index()
    st_stats.columns = ['racer_id', 'st_std_dev']
    st_stats['st_std_dev'] = st_stats['st_std_dev'].fillna(0.05)
    return st_stats


def load_synthetic_odds(conn, race_ids=None, start_date=None):
    """
    odds_dataから三連単オッズを使って各艇の「勝率(支持率)」を逆算する (高速単一パスクエリ)
    """
    print("Calculating Synthetic Odds (Fast Single-Pass)...")
    
    if start_date:
        query = f"""
        SELECT o.race_id, o.combination, o.odds_1min
        FROM odds_data o
        INNER JOIN races r ON o.race_id = r.race_id
        WHERE r.race_date >= '{start_date}' AND o.odds_1min > 0
        """
    elif race_ids:
        # race_idsが渡された場合、一時テーブルまたはIN句
        if len(race_ids) > 2000:
            query = "SELECT race_id, combination, odds_1min FROM odds_data WHERE odds_1min > 0"
        else:
            ids_str = "'" + "','".join(race_ids) + "'"
            query = f"SELECT race_id, combination, odds_1min FROM odds_data WHERE race_id IN ({ids_str}) AND odds_1min > 0"
    else:
        query = "SELECT race_id, combination, odds_1min FROM odds_data WHERE odds_1min > 0"
        
    try:
        df_odds = pd.read_sql(query, conn)
        if df_odds.empty:
            return None
        df_odds['odds_1min'] = pd.to_numeric(df_odds['odds_1min'], errors='coerce')
        df_odds = df_odds[df_odds['odds_1min'] > 0]
        df_odds['first_boat'] = df_odds['combination'].astype(str).str[0].astype(int)
        df_odds['prob'] = 1.0 / df_odds['odds_1min']
        syn = df_odds.groupby(['race_id', 'first_boat'])['prob'].sum().reset_index()
        syn.columns = ['race_id', 'boat_number', 'syn_win_rate']
        return syn
    except Exception as e:
        print(f"  Warning: Synthetic odds calculation error: {e}")
        return None


def process_wind_data(df):
    print("Processing Wind Vectors...")

    # 1. 風向テキストを角度(度数法: 北=0, 時計回り)に変換するマップ
    # ※「風が吹いてくる方角」
    direction_map = {
        '北': 0, '北東': 45, '東': 90, '南東': 135,
        '南': 180, '南西': 225, '西': 270, '北西': 315,
        # 表記ゆれや無風への対応
        '無風': np.nan, 'failed': np.nan, '': np.nan
    }

    # 2. 各レース場の「理想的な追い風（Tailwind）が吹いてくる方角」の定義
    venue_tailwind_from = {
        '桐生': 135,   # 1Mは北西 -> 南東風が追い風
        '戸田': 90,    # 1Mは西 -> 東風が追い風
        '江戸川': 180, # 1Mは北 -> 南風が追い風
        '平和島': 180, # 1Mは北 -> 南風が追い風
        '多摩川': 270, # 1Mは東 -> 西風が追い風
        '浜名湖': 180, # 1Mは北（やや北東?） -> 南風系統が追い風
        '蒲郡': 270,   # 1Mは東 -> 西風が追い風
        '常滑': 270,   # 1Mは東 -> 西風が追い風
        '津': 135,     # 1Mは北西 -> 南東風が追い風
        '三国': 180,   # 1Mは北 -> 南風が追い風
        'びわこ': 225, # 1Mは北東 -> 南西風が追い風
        '住之江': 270, # 1Mは東 -> 西風が追い風
        '尼崎': 90,    # 1Mは西 -> 東風が追い風
        '鳴門': 135,   # 1Mは北西 -> 南東風が追い風
        '丸亀': 180,   # 1Mは北 -> 南風が追い風
        '児島': 225,   # 1Mは北東 -> 南西風が追い風
        '宮島': 270,   # 1Mは東 -> 西風が追い風
        '徳山': 135,   # 1Mは北西 -> 南東風が追い風
        '下関': 270,   # 1Mは東 -> 西風が追い風
        '若松': 270,   # 1Mは東 -> 西風が追い風
        '芦屋': 135,   # 1Mは北西 -> 南東風が追い風
        '福岡': 0,     # 1Mは南 -> 北風が追い風
        '唐津': 135,   # 1Mは北西 -> 南東風が追い風
        '大村': 315    # 1Mは南東 -> 北西風が追い風
    }

    # 3. データのマッピング処理
    
    # 風向テキストを数値(Angle)に変換
    df['wind_angle_deg'] = df['wind_direction'].map(direction_map)
    
    # レース場ごとの追い風基準角をマッピング
    df['venue_tailwind_deg'] = df['venue_name'].map(venue_tailwind_from)

    # 欠損値（無風など）の処理
    df['wind_angle_deg'] = df['wind_angle_deg'].fillna(0)
    df['venue_tailwind_deg'] = df['venue_tailwind_deg'].fillna(0)
    df['wind_speed'] = df['wind_speed'].fillna(0)

    # 4. ベクトル計算 (Cos, Sin)
    # 角度差 = (風向 - 追い風基準)
    # ラジアンに変換
    angle_diff_rad = np.radians(df['wind_angle_deg'] - df['venue_tailwind_deg'])

    # wind_longitudinal (縦成分: 追い風/向かい風)
    df['wind_vector_long'] = df['wind_speed'] * np.cos(angle_diff_rad)

    # wind_lateral (横成分: 横風)
    df['wind_vector_lat'] = df['wind_speed'] * np.sin(angle_diff_rad)

    # 不要な一時カラムの削除
    df = df.drop(columns=['wind_angle_deg', 'venue_tailwind_deg'], errors='ignore')

    print("Wind processing complete.")
    return df

def process_features(df):
    print("Processing Features (including Cross Features & Momentum)...")
    
    # 欠損補正・数値型変換
    df['venue_code'] = pd.to_numeric(df['venue_code'], errors='coerce').fillna(1).astype(int)
    df['wind_speed'] = pd.to_numeric(df['wind_speed'], errors='coerce').fillna(0.0)
    df['wave_height'] = pd.to_numeric(df['wave_height'], errors='coerce').fillna(0.0)
    df['weight'] = pd.to_numeric(df['weight'], errors='coerce').fillna(52.0)
    df['nat_win_rate'] = pd.to_numeric(df['nat_win_rate'], errors='coerce').fillna(0.0)
    df['nat_quinella_rate'] = pd.to_numeric(df.get('nat_quinella_rate', 0.0), errors='coerce').fillna(0.0)
    df['local_win_rate'] = pd.to_numeric(df['local_win_rate'], errors='coerce').fillna(0.0)
    df['motor_rate'] = pd.to_numeric(df['motor_rate'], errors='coerce').fillna(30.0)
    df['boat_rate'] = pd.to_numeric(df['boat_rate'], errors='coerce').fillna(30.0)
    
    rank_map = {'A1': 4, 'A2': 3, 'B1': 2, 'B2': 1}
    df['racer_rank_num'] = df['racer_rank'].map(rank_map).fillna(2).astype(int)

    # 展示タイム欠損補完 (会場×艇番の中央値)
    df['exhibition_time'] = pd.to_numeric(df['exhibition_time'], errors='coerce')
    df['exhibition_time'] = df.groupby(['venue_code', 'boat_number'])['exhibition_time'].transform(lambda x: x.fillna(x.median())).fillna(6.80)

    # 決まり手出現率 (該当コースでの実績比率)
    denom = np.maximum(pd.to_numeric(df['course_run_count'], errors='coerce').fillna(0), 1.0)
    df['makuri_count'] = pd.to_numeric(df.get('makuri_count', 0), errors='coerce').fillna(0)
    df['makurizashi_count'] = pd.to_numeric(df.get('makurizashi_count', 0), errors='coerce').fillna(0)
    df['sashi_count'] = pd.to_numeric(df.get('sashi_count', 0), errors='coerce').fillna(0)
    df['nige_count'] = pd.to_numeric(df.get('nige_count', 0), errors='coerce').fillna(0)

    df['makuri_rate'] = df['makuri_count'] / denom
    df['makurizashi_rate'] = df['makurizashi_count'] / denom
    df['sashi_rate'] = df['sashi_count'] / denom
    df['nige_rate'] = df['nige_count'] / denom

    # --- ヘルパー関数: 今節平均着順のパース ---
    def parse_prior_results(res_str):
        if not isinstance(res_str, str): return np.nan
        ranks = [int(c) for c in res_str if c.isdigit()]
        if not ranks: return np.nan
        return np.mean(ranks)

    # 1. 今節平均着順 (Series Avg Rank)
    df['series_avg_rank'] = df['prior_results'].apply(parse_prior_results).fillna(3.5)

    # データをレースID, 予測コース(or枠番)順にソートしておく
    df = df.sort_values(['race_id', 'pred_course'])

    # A. ST関連 (Inner Gap, Slit Formation)
    df['inner_st'] = df.groupby('race_id')['exhibition_start_timing'].shift(1)
    df['inner_st_gap'] = (df['exhibition_start_timing'] - df['inner_st']).fillna(0.0)
    
    df['outer_st'] = df.groupby('race_id')['exhibition_start_timing'].shift(-1)
    avg_neighbor_st = (df['inner_st'].fillna(df['exhibition_start_timing']) + 
                       df['outer_st'].fillna(df['exhibition_start_timing'])) / 2
    df['slit_formation'] = (df['exhibition_start_timing'] - avg_neighbor_st).fillna(0.0)

    # B. 1マーク攻防 & 壁
    df['course1_nige_rate'] = df.groupby('race_id')['nige_rate'].transform('first')
    df['anti_nige_potential'] = (df['makuri_rate'] * (1.0 - df['course1_nige_rate'])).fillna(0.0)
    
    df['inner_quinella_rate'] = df.groupby('race_id')['course_quinella_rate'].shift(1)
    df['wall_strength'] = df['inner_quinella_rate'].fillna(0.0)
    
    df['inner_makuri_rate'] = df.groupby('race_id')['makuri_rate'].shift(1)
    df['follow_potential'] = (df['inner_makuri_rate'].fillna(0.0) * df['course_quinella_rate'].fillna(0.0))

    # C. 機力評価 & 代替モメンタム (Exhibition Momentum)
    # ① レース内相対展示タイム
    gb_tenji = df.groupby('race_id')['exhibition_time']
    race_min_ex = gb_tenji.transform('min')
    race_mean_ex = gb_tenji.transform('mean')
    race_std_ex = gb_tenji.transform('std')

    df['ex_diff_from_race_min'] = (df['exhibition_time'] - race_min_ex).fillna(0.0)
    df['ex_diff_from_race_mean'] = (df['exhibition_time'] - race_mean_ex).fillna(0.0)
    df['tenji_z_score'] = ((race_mean_ex - df['exhibition_time']) / race_std_ex.replace(0, np.nan)).fillna(0.0)
    df['ex_rank_in_race'] = gb_tenji.rank(method='min', ascending=True)
    df['linear_rank'] = df['ex_rank_in_race']
    df['is_linear_leader'] = (df['ex_rank_in_race'] == 1).astype(int)

    # ② 節間（同一会場・同一選手）での展示タイムモメンタム
    # 日付・レース番号順に並んでいる状態で shift
    df = df.sort_values(['race_date', 'race_id', 'pred_course'])
    df['prev_ex_in_series'] = df.groupby(['venue_code', 'racer_id'])['exhibition_time'].shift(1)
    df['ex_momentum_diff'] = (df['exhibition_time'] - df['prev_ex_in_series']).fillna(0.0)
    
    series_exp_mean = df.groupby(['venue_code', 'racer_id'])['exhibition_time'].transform(lambda x: x.expanding().mean())
    df['ex_momentum_deviation'] = (df['exhibition_time'] - series_exp_mean).fillna(0.0)

    # D. 風速クロス (Wind Speed Cross)
    df['is_strong_wind'] = (df['wind_speed'] >= 4.0).astype(float)
    df['is_gale_wind'] = (df['wind_speed'] >= 6.0).astype(float)
    df['wind_makuri_cross'] = df['wind_speed'] * df['makuri_rate']
    df['strong_wind_makuri'] = df['is_strong_wind'] * df['makuri_rate']
    df['wind_makurizashi_cross'] = df['wind_speed'] * df['makurizashi_rate']
    df['strong_wind_outer_adv'] = df['is_strong_wind'] * (df['boat_number'] >= 3).astype(float)
    df['wind_nige_vulnerability'] = df['wind_speed'] * (1.0 - df['nige_rate']) * (df['boat_number'] == 1).astype(float)
    df['high_wind_alert'] = (df['wind_speed'] >= 5.0).astype(int)

    # E. 波高クロス (Wave Height Cross)
    df['wave_weight_prod'] = df['wave_height'] * df['weight']
    df['wave_weight_ratio'] = df['wave_height'] / np.maximum(df['weight'], 40.0)
    df['is_high_wave'] = (df['wave_height'] >= 4.0).astype(float)
    df['high_wave_heavy_penalty'] = df['is_high_wave'] * np.maximum(0.0, df['weight'] - 52.0)
    df['high_wave_inner_risk'] = df['is_high_wave'] * (df['boat_number'] == 1).astype(float)

    # F. 環境・選手補正
    df['weight_diff'] = df['weight'] - df.groupby('race_id')['weight'].transform('mean')
    df['local_perf_diff'] = (df['local_win_rate'] - df['nat_win_rate']).fillna(0.0)

    # G. 風向ベクトル
    df = process_wind_data(df)

    # 不要な一時カラムの削除
    drop_cols = ['inner_st', 'outer_st', 'inner_quinella_rate', 'inner_makuri_rate', 'course1_nige_rate', 'prev_ex_in_series']
    df = df.drop(columns=drop_cols, errors='ignore')

    return df

def main():
    parser = argparse.ArgumentParser(description="Build Boatrace Dataset with Advanced Cross Features")
    parser.add_argument('--limit', type=int, default=None, help="Limit number of rows from DB for quick testing")
    parser.add_argument('--test', action='store_true', help="Run test mode with 6000 rows (1000 races)")
    parser.add_argument('--start_date', type=str, default=None, help="Filter races starting from this date (e.g. 2024-01-01)")
    parser.add_argument('--output', type=str, default='boatrace_dataset_labeled_v2.csv', help="Output CSV path")
    
    args = parser.parse_args()

    limit_count = args.limit
    if args.test and limit_count is None:
        limit_count = 6000

    conn = get_connection()
    
    # 1. ベースデータのロード
    print(f"Loading base data... (limit={limit_count}, start_date={args.start_date})")
    df = load_base_data(conn, limit=limit_count, start_date=args.start_date)
    
    if df.empty:
        print("❌ No data loaded.")
        conn.close()
        return df

    # 2. ST標準偏差の結合
    st_stats = load_st_stability(conn, limit=None)
    df = pd.merge(df, st_stats, on='racer_id', how='left')
    df['st_std_dev'] = df['st_std_dev'].fillna(0.05)
    
    # 3. オッズの結合 (合成オッズ)
    syn_odds = load_synthetic_odds(conn, start_date=args.start_date)

    
    if syn_odds is not None:
        df = pd.merge(df, syn_odds, on=['race_id', 'boat_number'], how='left')
        df['syn_win_rate'] = df['syn_win_rate'].fillna(0.0)
    else:
        print("Odds data not found or skipped.")
        df['syn_win_rate'] = 0.0
    
    # 4. 特徴量計算 (環境クロス・代替モメンタム含む)
    df_final = process_features(df)
    
    # 5. 確認 & 保存
    print("\n" + "=" * 70)
    print(f"  🎉 データセット作成完了: {len(df_final):,} 行 | {len(df_final.columns)} カラム")
    print("=" * 70)
    
    sample_cols = [
        'race_id', 'boat_number', 'wind_speed', 'wind_nige_vulnerability',
        'wind_makuri_cross', 'wave_weight_prod', 'high_wave_heavy_penalty',
        'ex_diff_from_race_min', 'ex_momentum_diff', 'series_avg_rank', 'rank'
    ]
    avail_sample_cols = [c for c in sample_cols if c in df_final.columns]
    print(df_final[avail_sample_cols].head(6))
    
    # 保存
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_final.to_csv(args.output, index=False)
    print(f"\nSaved dataset to: {args.output}")

    
    conn.close()
    return df_final

if __name__ == "__main__":
    df_dataset = main()