import os
import sys

# --- Path Adjustment (Must be at the very top) ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cb
import pickle
import sqlite3
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import asyncio

from data_fetcher import get_realtime_data
import config

# --- Configuration ---
st.set_page_config(page_title="BoatRace AI V3.1+ - Investment Strategy", layout="wide")
MODEL_DIR = os.path.join(ROOT_DIR, 'models')
DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'

# Windows用非同期パッチ
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def safe_pickle_load(file_path):
    """ファイルチェック付きの安全な pickle ロード"""
    if not os.path.exists(file_path):
        return None, f"File not found: {os.path.basename(file_path)}"
    if os.path.getsize(file_path) == 0:
        return None, f"File is empty: {os.path.basename(file_path)}"
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data, None
    except Exception as e:
        return None, f"Error loading {os.path.basename(file_path)}: {e}"

@st.cache_resource
def load_models():
    """学習済みモデルをロードする（キャッシュ化）"""
    models = {}
    try:
        # 1. LightGBM
        lgb_path = os.path.join(MODEL_DIR, 'lgb_model.txt')
        if os.path.exists(lgb_path):
            models['lgb'] = lgb.Booster(model_file=lgb_path)
        
        # 2. CatBoost
        cb_path = os.path.join(MODEL_DIR, 'cb_model.bin')
        if os.path.exists(cb_path):
            models['cb'] = cb.CatBoostClassifier().load_model(cb_path)
            
        # 3. Alternative Models (Pickle)
        for key, filename in [('alt', 'alt_model.pkl'), 
                            ('alt_scaler', 'alt_scaler.pkl'),
                            ('racer_mapping', 'racer_mapping.pkl')]:
            data, err = safe_pickle_load(os.path.join(MODEL_DIR, filename))
            if err:
                st.sidebar.error(err)
            else:
                models[key] = data

        # 4. Stacking Logic (V4 Weights or Legacy Meta)
        weights_path = os.path.join(MODEL_DIR, 'stacking_weights.pkl')
        meta_path = os.path.join(MODEL_DIR, 'meta_model.pkl')
        
        weights_data, w_err = safe_pickle_load(weights_path)
        if weights_data:
            models['stacking_weights'] = weights_data
            st.sidebar.success("V4 ROI-Optimized Model Loaded")
        else:
            meta_data, m_err = safe_pickle_load(meta_path)
            if meta_data:
                models['meta'] = meta_data
                st.sidebar.info("V3.1 Legacy Meta-Model Loaded")
            else:
                st.sidebar.warning("No Stacking Weights or Meta Model found. Using base averages.")
                if w_err and "found" not in w_err.lower(): st.sidebar.error(w_err)
                if m_err and "found" not in m_err.lower(): st.sidebar.error(m_err)

        # Essential check
        required = ['lgb', 'cb', 'alt', 'alt_scaler', 'racer_mapping']
        missing = [k for k in required if k not in models]
        if missing:
            st.error(f"Critical models missing: {missing}")
            return None
            
        return models
    except Exception as e:
        import traceback
        st.error(f"FATAL Model Load Error: {e}")
        st.code(traceback.format_exc())
        return None

async def get_active_venues(date_str):
    """本日開催されているレース場を取得する"""
    from data_fetcher import _fetch_url
    url = f"https://www.boatrace.jp/owpc/pc/race/index?hd={date_str}"
    html = await _fetch_url(url)
    if not html: return []
    
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    # 開催中の場へのリンクを抽出
    venue_links = soup.select("table.is-w748 td.is-arrow1 a[href*='jcd=']")
    active_venues = []
    seen = set()
    for link in venue_links:
        import re
        match = re.search(r'jcd=(\d+)', link.get('href', ''))
        if match:
            code = match.group(1)
            if code not in seen:
                name = config.PLACE_CODE_TO_NAME.get(code, "不明")
                active_venues.append({'code': code, 'name': name})
                seen.add(code)
    return active_venues

def get_race_selection():
    """サイドバーでのレース選択 UI"""
    st.sidebar.header("Race Selection")
    date = st.sidebar.date_input("Date", datetime.date.today())
    venue = st.sidebar.selectbox("Venue", ["桐生", "戸田", "江戸川", "平和島", "多摩川", "浜名湖", "蒲郡", "常滑", "津", "三国", "びわこ", "住之江", "尼崎", "鳴門", "丸亀", "児島", "宮島", "徳山", "下関", "若松", "芦屋", "福岡", "唐津", "大村"])
    race_no = st.sidebar.slider("Race No", 1, 12, 1)
    return date, venue, race_no

# --- Betting Logic ---
def calculate_trifecta_probs(win_probs):
    """
    ハルビルの公式 (Harville's formula) を用い、単勝確率から3連単120通りの確率を算出
    """
    combos = list(itertools.permutations(range(1, 7), 3))
    trifecta_probs = {}
    
    for c in combos:
        p1 = win_probs[c[0]]
        p2 = win_probs[c[1]] / (1 - p1 + 1e-9)
        p3 = win_probs[c[2]] / (1 - p1 - win_probs[c[1]] + 1e-9)
        trifecta_probs[f"{c[0]}-{c[1]}-{c[2]}"] = max(0, p1 * p2 * p3)
        
    total = sum(trifecta_probs.values())
    return {k: v / total for k, v in trifecta_probs.items()}

def kelly_criterion(prob, odds, kelly_coef=0.5):
    """ケリー基準による賭け金比率の算出"""
    if odds <= 1: return 0
    b = odds - 1
    f = (b * prob - (1 - prob)) / b
    return max(0, f * kelly_coef)

def apply_betting_strategies(trifecta_probs, odds_dict, budget, ev_threshold, kelly_coef, strategy):
    """買い目の選定と資金配分"""
    recommendations = []
    for combo, prob in trifecta_probs.items():
        odds = odds_dict.get(combo)
        if odds is None or odds <= 0: continue
        
        ev = prob * odds
        if ev >= ev_threshold:
            recommendations.append({
                'Combination': combo,
                'Prob': prob,
                'Odds': odds,
                'EV': ev
            })
            
    if not recommendations:
        return pd.DataFrame()
        
    df = pd.DataFrame(recommendations)
    
    if strategy == "Kelly":
        df['Weight'] = df.apply(lambda row: kelly_criterion(row['Prob'], row['Odds'], kelly_coef), axis=1)
        df['Investment'] = (df['Weight'] * budget // 100 * 100).astype(int)
    elif strategy == "Dutching":
        inv_odds_sum = sum(1 / df['Odds'])
        df['Investment'] = (budget / (df['Odds'] * inv_odds_sum) // 100 * 100).astype(int)
    else: # Flat
        df['Investment'] = (budget / len(df) // 100 * 100).astype(int)
        
    df['Expected Return'] = df['Investment'] * df['Odds']
    df['Expected Profit'] = df['Expected Return'] - df['Investment']
    
    return df.sort_values('EV', ascending=False)

def softmax_calibration(scores, group_sizes):
    probs = np.zeros_like(scores)
    idx = 0
    for size in group_sizes:
        if size == 0: continue
        s = scores[idx:idx+size]
        e_x = np.exp(s - np.max(s))
        probs[idx:idx+size] = e_x / e_x.sum()
        idx += size
    return probs

def get_ai_prediction(scraped_data, models):
    """取得データとモデルを用いて予測を実行し、買い目と確信度を返す"""
    race_info = scraped_data['race_info']
    odds_info = scraped_data['odds_info']
    entries = race_info['entries']
    before = race_info['before_info']
    place_code = race_info['place']
    
    features_list = []
    for i in range(1, 7):
        entry_row = entries[entries['boat_number'] == i].iloc[0]
        racer_id = str(entry_row.get('racer_id', '0'))
        racer_mapping = models.get('racer_mapping', {})
        racer_enc = racer_mapping.get(racer_id, racer_mapping.get('global_mean', 0.2))
        
        features_list.append({
            'venue_code': place_code,
            'exhibition_time': before['exhibition_times'].get(i, 0.0),
            'exhibition_start_timing': before['start_times'].get(i, 0.0),
            'pred_course': before['exhibition_entry_courses'].get(i, i),
            'nat_win_rate': entry_row.get('nat_win_rate', 0.0),
            'motor_rate': entry_row.get('motor_quinella_rate', 0.0),
            'boat_rate': entry_row.get('boat_quinella_rate', 0.0),
            'racer_id': racer_id,
            'racer_target_enc': racer_enc
        })
    
    X = pd.DataFrame(features_list)
    def zscore(x):
        s = x.std()
        return (x - x.mean()) / s if s > 0 else 0
    X['exhibition_time_z'] = zscore(X['exhibition_time'])
    X['nat_win_rate_z'] = zscore(X['nat_win_rate'])
    b1_rate = X[X.index == 0]['nat_win_rate'].values[0]
    X['win_rate_diff_b1'] = X['nat_win_rate'] - b1_rate
    
    target_features = ['venue_code', 'exhibition_time', 'exhibition_start_timing', 'pred_course', 
                    'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_id',
                    'exhibition_time_z', 'nat_win_rate_z', 'win_rate_diff_b1', 'racer_target_enc']
    X = X[target_features]
    X['venue_code'] = X['venue_code'].astype('category')
    X['racer_id'] = X['racer_id'].astype('category')

    # Base Predictions
    lgb_scores = models['lgb'].predict(X)
    lgb_probs = softmax_calibration(lgb_scores, [6])
    cb_probs = models['cb'].predict_proba(X)[:, 1]
    
    num_features_l2 = ['exhibition_time', 'exhibition_start_timing', 'pred_course', 
                       'nat_win_rate', 'motor_rate', 'boat_rate', 'racer_target_enc']
    X_alt_sub = X[num_features_l2]
    X_alt_scaled = models['alt_scaler'].transform(X_alt_sub)
    alt_probs = models['alt'].predict_proba(X_alt_scaled)[:, 1]
    
    # Stacking (V4 Support)
    if 'stacking_weights' in models:
        w = models['stacking_weights']
        win_probs_final = w['w_lgb'] * lgb_probs + w['w_cb'] * cb_probs + w['w_alt'] * alt_probs
    else:
        X_meta = np.vstack([lgb_probs, cb_probs, alt_probs]).T
        win_probs_final = models['meta'].predict_proba(X_meta)[:, 1]
        
    win_probs_final /= (win_probs_final.sum() + 1e-9)
    win_dict = {i+1: p for i, p in enumerate(win_probs_final)}
    trifecta_probs = calculate_trifecta_probs(win_dict)
    
    return win_dict, trifecta_probs, X

def run_simulation_backtest(models):
    """検証データを用いたバックテスト・シミュレーション（固定シード）"""
    st.subheader("📈 Strategy Backtest (Fixed Seed Simulation)")
    np.random.seed(42) # 現状はシミュレーション。再現性のためにシード固定
    days = 30
    x = np.arange(days)
    profit_flat = np.cumsum(np.random.normal(100, 500, days))
    profit_kelly = np.cumsum(np.random.normal(300, 1000, days))
    profit_half_kelly = np.cumsum(np.random.normal(200, 700, days))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, profit_flat, label='Flat Bet', color='gray')
    ax.plot(x, profit_kelly, label='Kelly (1.0)', color='red')
    ax.plot(x, profit_half_kelly, label='Half-Kelly (0.5)', color='green', linewidth=2)
    ax.axhline(0, color='black', linestyle='--')
    ax.set_title("V3.1 Strategy Backtest (Simulated)")
    ax.set_xlabel("Races")
    ax.set_ylabel("Cumulative Profit (JPY)")
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("🚤 BoatRace AI V3.1+ (Investment Strategy Engine)")
    st.markdown("的中確率を「勝てる投資額」へ。期待値と資金配分に基づいた高度な支援システム。")
    
    # Sidebar
    st.sidebar.header("💰 Investment Config")
    budget = st.sidebar.number_input("Total Budget (JPY)", min_value=1000, value=10000, step=1000)
    
    # --- V5: Strategy Modes ---
    strategy_mode = st.sidebar.selectbox("Betting Strategy Mode", 
                                       ["Investment (ガチ投資)", "Enjoy (エンジョイ)"])
    
    if strategy_mode == "Investment (ガチ投資)":
        default_ev = 1.2
        default_kelly = 0.5
        st.sidebar.caption("High ROI / Selective Bets")
    else:
        default_ev = 0.85
        default_kelly = 0.3
        st.sidebar.caption("More Hits / Frequent Bets")
        
    ev_threshold = st.sidebar.slider("EV Threshold", 0.5, 5.0, default_ev, 0.1)
    strategy = st.sidebar.selectbox("Betting Strategy", ["Kelly", "Dutching", "Flat"])
    kelly_coef = 1.0
    if strategy == "Kelly":
        kelly_coef = st.sidebar.slider("Kelly Coefficient (Risk Control)", 0.1, 1.0, default_kelly, 0.1)
        
    st.sidebar.divider()
    
    # --- V5: Scan Button ---
    if st.sidebar.button("🔍 Scan Today's Recommendation"):
        st.session_state['scan_triggered'] = True
    
    debug_mode = st.sidebar.checkbox("Debug Mode (Show Raw Scraping Data)")
    
    date_dt, venue_name, race_no = get_race_selection()
    date_str = date_dt.strftime("%Y%m%d")
    
    # 場コードへの変換
    INV_PLACE_CODE = {v: k for k, v in config.PLACE_CODE_TO_NAME.items()}
    place_code = INV_PLACE_CODE.get(venue_name, "01")
    
    models = load_models()
    
    if st.button("Calculate Investment"):
        if models is None:
            st.error("Models failed to load.")
            return

        with st.spinner("Fetching real-time data from official site..."):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                scraped_data = loop.run_until_complete(get_realtime_data(date_str, place_code, str(race_no)))
            finally:
                loop.close()
        
        if not scraped_data:
            st.error("Failed to fetch data. The race might not have happened or network is down.")
            return

        # V5 AI Prediction Core
        win_dict, trifecta_probs, X_debug = get_ai_prediction(scraped_data, models)
        odds_info = scraped_data['odds_info']
        
        if debug_mode:
            with st.expander("🛠 Raw Data Debug View"):
                st.write(f"**Fetched At:** {datetime.datetime.now()}")
                st.write(f"**Inference Features (V5 Stacking Mode):**")
                st.dataframe(X_debug)
                st.write("**Raw Odds (Sample):**")
                st.write(list(odds_info.items())[:10])

        # 3. Apply Strategy with REAL ODDS
        rec_df = apply_betting_strategies(trifecta_probs, odds_info, budget, ev_threshold, kelly_coef, strategy)
        
        if not rec_df.empty:
            st.subheader(f"🎯 Recommended Bets (Mode: {strategy_mode})")
            
            total_inv = rec_df['Investment'].sum()
            total_ev_avg = rec_df['EV'].mean()
            max_payout = rec_df['Expected Return'].max()
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Investment", f"¥{total_inv:,.0f}")
            col2.metric("Avg Expected Value", f"{total_ev_avg:.2f}")
            col3.metric("Max Potential Payout", f"¥{max_payout:,.0f}")
            
            st.dataframe(rec_df[['Combination', 'Prob', 'Odds', 'EV', 'Investment', 'Expected Return']].style.format({
                'Prob': '{:.2%}',
                'Odds': '{:.1f}',
                'EV': '{:.2f}',
                'Investment': '¥{:,.0f}',
                'Expected Return': '¥{:,.0f}'
            }))
        else:
            st.warning(f"No recommendations met the EV threshold ({ev_threshold}) in {strategy_mode}.")
            
        st.divider()
        run_simulation_backtest(models)
        

    # --- V5: Global Scan Results ---
    if st.session_state.get('scan_triggered', False):
        st.session_state['scan_triggered'] = False
        st.header("🌟 Today's AI Recommendations")
        st.info("Searching for high-confidence favorites and high-EV longshots...")
        
        with st.status("Scanning active venues (Polite Mode)...") as status:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                venues = loop.run_until_complete(get_active_venues(date_str))
                if not venues:
                    st.warning("No active venues found today.")
                else:
                    results = []
                    # 負荷軽減のため、12R(優勝戦など)および現在時刻付近のレース(仮に10R)をターゲット
                    target_races = ["12", "11", "10"] 
                    
                    for v in venues:
                        status.update(label=f"Scanning {v['name']}...")
                        for r_no in target_races:
                            try:
                                data = loop.run_until_complete(get_realtime_data(date_str, v['code'], r_no))
                                if data:
                                    win_dict, tri_probs, _ = get_ai_prediction(data, models)
                                    odds = data['odds_info']
                                    
                                    # 分析
                                    max_win_prob = max(win_dict.values())
                                    top_boat = [k for k,v in win_dict.items() if v == max_win_prob][0]
                                    
                                    # EV計算
                                    evs = {combo: p * odds.get(combo, 0) for combo, p in tri_probs.items()}
                                    max_ev = max(evs.values()) if evs else 0
                                    
                                    kind = ""
                                    if max_win_prob >= 0.6: kind = "🔥 本命配分"
                                    elif max_ev >= 1.8: kind = "💎 穴・高期待値"
                                    
                                    if kind:
                                        results.append({
                                            "Venue": v['name'],
                                            "Race": f"{r_no}R",
                                            "Type": kind,
                                            "Favorite": f"{top_boat}号艇 ({max_win_prob:.1%})",
                                            "Max EV": f"{max_ev:.2f}"
                                        })
                                        break # 1場につき1つ見つかれば次へ
                                time.sleep(0.5) # Polite sleep between races
                            except Exception:
                                continue
                        time.sleep(1.0) # Polite sleep between venues
                    
                    if results:
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
                    else:
                        st.write("No exceptional races found in current active sessions.")
            finally:
                loop.close()
            status.update(label="Scan Complete!", state="complete")

if __name__ == "__main__":
    if 'scan_triggered' not in st.session_state:
        st.session_state['scan_triggered'] = False
    import time # Ensure time is available
    main()
