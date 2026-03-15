import sqlite3
import pandas as pd
import numpy as np
import re

# データベースパス (適宜変更してください)
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db' 

def get_connection():
    return sqlite3.connect(DB_PATH)

def load_base_data(conn, limit=None):
    """
    基本データと統計データを結合して取得する
    limit: テスト用に取得件数を制限する場合に指定
    """
    query = """
    SELECT
        -- 識別子
        re.race_id,
        re.boat_number,
        re.racer_id,
        r.race_date,
        r.venue_code,
        
        -- 直前情報 & 進入予想
        -- 展示進入がない場合は枠番を使用するCOALESCE処理
        bi.exhibition_time,
        bi.exhibition_start_timing,
        COALESCE(bi.exhibition_entry_course, re.boat_number) as pred_course,
        
        -- 選手能力 (絶対評価)
        re.nat_win_rate,
        re.motor_rate,
        re.boat_rate,
        re.prior_results,
        re.weight,
        re.branch,
        
        -- レース結果 (rank) from results
        res.finish_order as rank,
        
        -- コース別成績 (Racer_CourseStats)
        rcs.RacesRun as course_run_count,
        rcs.QuinellaRate as course_quinella_rate,
        rcs.TrifectaRate as course_trifecta_rate,
        rcs.FirstPlaceRate as course_1st_rate,
        rcs.AvgStartTiming as course_avg_st,
        
        -- 決まり手実績 (Racer_CourseWinTech)
        -- ※NULLの場合は0埋め
        COALESCE(wt.Nige, 0) as nige_count,
        COALESCE(wt.Makuri, 0) as makuri_count,
        COALESCE(wt.Sashi, 0) as sashi_count,
        
        -- 会場・環境情報
        r.wind_speed,
        r.wind_direction,
        r.wave_height,
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
        
    -- 決まり手の結合
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
    
    ORDER BY re.race_id DESC, re.boat_number
    """
    
    if limit:
        query += f" LIMIT {limit}"
        
    return pd.read_sql(query, conn)

def load_st_stability(conn, limit=None):
    """
    resultsテーブルから直近のST標準偏差を計算する
    """
    print("Calculating ST Standard Deviation from results...")
    # 全期間だと重いので、直近6ヶ月などに絞るのが一般的ですが、今回は全件で例示
    query = "SELECT racer_id, start_timing FROM results"
    if limit:
        query += f" LIMIT {limit}"
    df_res = pd.read_sql(query, conn)
    
    # 欠損除去
    df_res = df_res.dropna()
    
    # 計算 (選手ごとのST標準偏差)
    st_stats = df_res.groupby('racer_id')['start_timing'].std().reset_index()
    st_stats.columns = ['racer_id', 'st_std_dev']
    
    # NaN(1走のみ等)は平均的な値(0.05程度)で埋める
    st_stats['st_std_dev'] = st_stats['st_std_dev'].fillna(0.05)
    
    return st_stats

def load_synthetic_odds(conn, race_ids):
    """
    odds_dataから三連単オッズを使って各艇の「勝率(支持率)」を逆算する
    odds_data table uses 'combination' (e.g. '123' or '1-2-3') instead of first_boat.
    """
    print("Calculating Synthetic Odds...")
    # 対象レースのみ取得
    ids_str = "'" + "','".join(race_ids) + "'"
    query = f"""
    SELECT race_id, combination, odds_1min 
    FROM odds_data 
    WHERE race_id IN ({ids_str})
    """
    try:
        df_odds = pd.read_sql(query, conn)
    except:
        return None # オッズデータがない場合

    # Ensure numeric
    df_odds['odds_1min'] = pd.to_numeric(df_odds['odds_1min'], errors='coerce')
    
    # Parse 'first_boat' from 'combination'
    # Combination format: '123' or '1-2-3'. Assume 1st char is the first boat.
    # Note: If '10' exists (not in boat race usually, max 6), logic differs.
    # But boat race is 1-6. So str[0] is safe.
    df_odds['first_boat'] = df_odds['combination'].astype(str).str[0].astype(int)

    # オッズの逆数（支持率）を計算
    # 0.0や欠損を除く
    df_odds = df_odds[df_odds['odds_1min'] > 0].copy()
    df_odds['prob'] = 1 / df_odds['odds_1min']
    
    # 各レース・各艇ごとの「1着になる確率の合計」を出す
    syn_odds = df_odds.groupby(['race_id', 'first_boat'])['prob'].sum().reset_index()
    syn_odds.columns = ['race_id', 'boat_number', 'syn_win_rate']
    
    # 正規化（合計が1になるわけではない控除率の影響があるため、レースごとにスケーリングしても良いが、ここではそのままで）
    return syn_odds

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
    print("Processing Features...")
    
    # --- ヘルパー関数: 今節平均着順のパース ---
    def parse_prior_results(res_str):
        if not isinstance(res_str, str): return np.nan
        # 数字のみ抽出
        ranks = [int(c) for c in res_str if c.isdigit()]
        # F/L等の事故は文字として残るが、ここでは簡略化のため数字のみの平均
        # 厳密にするなら: re.findall(r'[FL]', res_str) で事故数をカウントし、6点(最下位)として加算など
        if not ranks: return np.nan
        return np.mean(ranks)

    # 1. 今節平均着順 (Series Avg Rank)
    df['series_avg_rank'] = df['prior_results'].apply(parse_prior_results)
    # 欠損は3.5(中間)で埋める
    df['series_avg_rank'] = df['series_avg_rank'].fillna(3.5)

    # 2. まくり率・逃げ率の正規化 (回数 -> 率)
    # 分母が0の場合は0にする
    df['makuri_rate'] = df['makuri_count'] / df['course_run_count'].replace(0, 1)
    df['nige_rate'] = df['nige_count'] / df['course_run_count'].replace(0, 1)

    # --- 相対・グループ特徴量の計算 ---
    # レースIDでグルーピングして計算する
    
    # データをレースID, 予測コース(or枠番)順にソートしておく
    df = df.sort_values(['race_id', 'pred_course'])
    
    # A. ST関連 (Inner Gap, Slit Formation)
    # 内隣のSTを取得 (shift 1)
    df['inner_st'] = df.groupby('race_id')['exhibition_start_timing'].shift(1)
    
    # ① 対内隣ST差 (マイナスが良い)
    # 1コースは内隣がいないので0とする
    df['inner_st_gap'] = df['exhibition_start_timing'] - df['inner_st']
    df['inner_st_gap'] = df['inner_st_gap'].fillna(0)
    
    # 外隣のST (shift -1)
    df['outer_st'] = df.groupby('race_id')['exhibition_start_timing'].shift(-1)
    
    # ② スリット隊形係数 (自分が凹んで両隣が出ているか)
    # (自ST) - (内と外の平均)
    avg_neighbor_st = (df['inner_st'].fillna(df['exhibition_start_timing']) + 
                       df['outer_st'].fillna(df['exhibition_start_timing'])) / 2
    df['slit_formation'] = df['exhibition_start_timing'] - avg_neighbor_st

    # B. 1マーク攻防
    
    # ④ イン逃げ阻止力 (Anti-Nige)
    # 自分がまくり屋(makuri_rate)で、かつ1コースが逃げ失敗しやすいか？
    # まず1コースのデータを全艇に持たせる
    df['course1_nige_rate'] = df.groupby('race_id')['nige_rate'].transform('first')
    # 1コース以外が持つ特徴量
    df['anti_nige_potential'] = df['makuri_rate'] * (1 - df['course1_nige_rate'])
    
    # ⑤ 壁信頼度 (Wall Strength)
    # 2コースの艇の連対率を、そのレースの全艇に「壁強度」として配る(あるいは2コースの選手自身のFeatureとする)
    # ここでは「自艇の一つ内側が壁として機能するか」を計算します
    df['inner_quinella_rate'] = df.groupby('race_id')['course_quinella_rate'].shift(1)
    df['wall_strength'] = df['inner_quinella_rate'] # シンプルに内側の強さ
    
    # ⑥ 追随ポテンシャル (Follow Potential)
    # 内隣がまくり屋なら、自分にチャンス
    df['inner_makuri_rate'] = df.groupby('race_id')['makuri_rate'].shift(1)
    df['follow_potential'] = df['inner_makuri_rate'] * df['course_quinella_rate']
    
    # C. 機力評価
    
    # ⑬ 展示タイム偏差値 (Tenji Z-Score)
    # レースごとの平均と標準偏差
    gb_tenji = df.groupby('race_id')['exhibition_time']
    df['tenji_mean'] = gb_tenji.transform('mean')
    df['tenji_std'] = gb_tenji.transform('std')
    # 偏差値化 (タイムは小さい方が良いので符号反転するか、 (平均 - 自分) / std とする)
    df['tenji_z_score'] = (df['tenji_mean'] - df['exhibition_time']) / df['tenji_std']
    df['tenji_z_score'] = df['tenji_z_score'].fillna(0) # stdが0(全員同じ)の場合など

    # ⑭ 直線番長フラグ (Rank)
    df['linear_rank'] = gb_tenji.rank(method='min', ascending=True) # タイム昇順のランク
    df['is_linear_leader'] = (df['linear_rank'] == 1).astype(int)

    # D. 環境・選手補正
    
    # ⑩ 体重ハンデ
    # レース平均体重との差
    df['weight_diff'] = df['weight'] - df.groupby('race_id')['weight'].transform('mean')
    
    # ⑲ イン有利風向 (単純化: 風向データがテキストの場合、パースが必要)
    # 例: "北西" などの場合、場ごとの地理データが必要ですが、
    # ここでは「追い風フラグ」などはwind_directionの処理が複雑なため、
    # 簡易的に「風速の影響」のみ実装します
    df['high_wind_alert'] = (df['wind_speed'] >= 5).astype(int)
    
    # ⑳ 当地相性
    df['local_perf_diff'] = df['local_win_rate'] - df['nat_win_rate']
    df['local_perf_diff'] = df['local_perf_diff'].fillna(0)

    # 不要な一時カラムの削除
    drop_cols = ['inner_st', 'outer_st', 'inner_quinella_rate', 'inner_makuri_rate', 
                 'tenji_mean', 'tenji_std', 'course1_nige_rate']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Wind processing
    df = process_wind_data(df)
    
    return df

def main():
    conn = get_connection()
    
    # 1. ベースデータのロード
    print("Loading base data...")
    # 1. ベースデータのロード
    print("Loading base data... (ALL)")
    df = load_base_data(conn, limit=None) # 全件取得
    
    # 2. ST標準偏差の結合
    st_stats = load_st_stability(conn, limit=None) # 全件取得
    df = pd.merge(df, st_stats, on='racer_id', how='left')
    
    # 3. オッズの結合 (オプション)
    # 対象レースIDのリストを取得
    unique_race_ids = df['race_id'].unique().tolist()
    syn_odds = load_synthetic_odds(conn, unique_race_ids)
    
    if syn_odds is not None:
        df = pd.merge(df, syn_odds, on=['race_id', 'boat_number'], how='left')
        df['syn_win_rate'] = df['syn_win_rate'].fillna(0)
    else:
        print("Odds data not found or skipped.")
        df['syn_win_rate'] = 0
    
    # 4. 特徴量計算
    df_final = process_features(df)
    
    # 5. 確認
    print("Data creation complete.")
    print(df_final[['race_id', 'boat_number', 'inner_st_gap', 'tenji_z_score', 'series_avg_rank']].head(12))
    
    # 保存
    df_final.to_csv('boatrace_dataset_labeled_v2.csv', index=False)
    
    conn.close()
    return df_final

if __name__ == "__main__":
    df_dataset = main()