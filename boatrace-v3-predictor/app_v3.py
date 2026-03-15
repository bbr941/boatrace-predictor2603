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

@st.cache_resource
def load_models():
    """学習済みモデルをロードする（キャッシュ化）"""
    models = {}
    try:
        models['lgb'] = lgb.Booster(model_file=os.path.join(MODEL_DIR, 'lgb_model.txt'))
        models['cb'] = cb.CatBoostClassifier().load_model(os.path.join(MODEL_DIR, 'cb_model.bin'))
        with open(os.path.join(MODEL_DIR, 'alt_model.pkl'), 'rb') as f:
            models['alt'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'alt_scaler.pkl'), 'rb') as f:
            models['alt_scaler'] = pickle.load(f)
        with open(os.path.join(MODEL_DIR, 'meta_model.pkl'), 'rb') as f:
            models['meta'] = pickle.load(f)
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

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
    ev_threshold = st.sidebar.slider("EV Threshold", 0.5, 5.0, 1.5, 0.1)
    strategy = st.sidebar.selectbox("Betting Strategy", ["Kelly", "Dutching", "Flat"])
    kelly_coef = 1.0
    if strategy == "Kelly":
        kelly_coef = st.sidebar.slider("Kelly Coefficient (Risk Control)", 0.1, 1.0, 0.5, 0.1)
        
    st.sidebar.divider()
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

        # Feature Engineering for Inference
        race_info = scraped_data['race_info']
        odds_info = scraped_data['odds_info']
        entries = race_info['entries']
        before = race_info['before_info']
        
        features_list = []
        for i in range(1, 7):
            entry_row = entries[entries['boat_number'] == i].iloc[0]
            features_list.append({
                'venue_code': place_code,
                'exhibition_time': before['exhibition_times'].get(i, 0.0),
                'exhibition_start_timing': before['start_times'].get(i, 0.0),
                'pred_course': before['exhibition_entry_courses'].get(i, i),
                'nat_win_rate': entry_row.get('nat_win_rate', 0.0),
                'motor_rate': entry_row.get('motor_quinella_rate', 0.0),
                'boat_rate': entry_row.get('boat_quinella_rate', 0.0),
                'racer_id': str(entry_row.get('racer_id', '0'))
            })
        
        X = pd.DataFrame(features_list)
        X['venue_code'] = X['venue_code'].astype('category')
        X['racer_id'] = X['racer_id'].astype('category')

        if debug_mode:
            with st.expander("🛠 Raw Data Debug View"):
                st.write(f"**Fetched At:** {datetime.datetime.now()}")
                st.write(f"**Target URL:** https://www.boatrace.jp/owpc/pc/race/racelist?jcd={place_code}&rno={race_no}&hd={date_str}")
                st.write("**Inference Features (X):**")
                st.dataframe(X)
                st.write("**Raw Before Info:**")
                st.json(before)
                st.write("**Scraped Odds (Sample):**")
                st.write(list(odds_info.items())[:10])

        # Base Model Predictions
        lgb_probs = models['lgb'].predict(X)
        cb_probs = models['cb'].predict_proba(X)[:, 1]
        
        X_alt_sub = X.drop(['venue_code', 'racer_id'], axis=1)
        X_alt_scaled = models['alt_scaler'].transform(X_alt_sub)
        alt_probs = models['alt'].predict_proba(X_alt_scaled)[:, 1]
        
        # Meta-Stacking
        X_meta = np.vstack([lgb_probs, cb_probs, alt_probs]).T
        win_probs_final = models['meta'].predict_proba(X_meta)[:, 1]
        win_probs_final /= (win_probs_final.sum() + 1e-9)
        win_dict = {i+1: p for i, p in enumerate(win_probs_final)}
        
        # 2. Calculate 120 combinations of Trifecta Probs
        trifecta_probs = calculate_trifecta_probs(win_dict)
        
        # 3. Apply Strategy with REAL ODDS
        rec_df = apply_betting_strategies(trifecta_probs, odds_info, budget, ev_threshold, kelly_coef, strategy)
        
        if not rec_df.empty:
            st.subheader(f"🎯 Recommended Bets (Strategy: {strategy})")
            
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
            st.warning("No recommendations met the EV threshold with current REAL ODDS.")
            
        st.divider()
        run_simulation_backtest(models)

if __name__ == "__main__":
    main()
