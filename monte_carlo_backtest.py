import pandas as pd
import numpy as np
import sqlite3
import itertools
import optuna
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from tqdm import tqdm

DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'

class RealDataBacktester:
    def __init__(self, limit_races=100):
        self.limit_races = limit_races
        self.features = [
            'nat_win_rate', 'loc_win_rate', 'motor_rate', 
            'exhibition_time', 'wind_speed'
        ]
        self.data = self.load_real_data()
        self.scaler = MinMaxScaler()
        
    def load_real_data(self):
        print(f"Loading data from {DB_PATH} (Limit: {self.limit_races} races)...")
        conn = sqlite3.connect(DB_PATH)
        
        # モンスターDB(8GB)でのJOINを避ける工夫
        # 首先获取 race_ids
        race_ids_df = pd.read_sql("SELECT race_id FROM races WHERE race_date > '2024-10-01' ORDER BY race_date DESC LIMIT ?", 
                                  conn, params=(self.limit_races,))
        race_ids = tuple(race_ids_df['race_id'].tolist())
        
        # 最小限のJOIN
        query_filtered = f"""
        SELECT 
            r.race_id, r.race_date, r.wind_speed,
            e.boat_number, e.nat_win_rate, e.loc_win_rate, e.motor_rate,
            b.exhibition_time,
            p.sanrentan_result, p.sanrentan_payoff
        FROM races r
        JOIN race_entries e ON r.race_id = e.race_id
        JOIN before_info b ON r.race_id = b.race_id AND e.boat_number = b.boat_number
        JOIN payoffs p ON r.race_id = p.race_id
        WHERE r.race_id IN {race_ids}
        """
        
        df = pd.read_sql(query_filtered, conn)
        conn.close()
        
        # 欠損値補完
        df['exhibition_time'] = df['exhibition_time'].fillna(df['exhibition_time'].mean())
        
        return df

    def preprocess(self):
        print("Preprocessing and Normalizing features...")
        self.data[self.features] = self.scaler.fit_transform(self.data[self.features])
        
        self.races = []
        for race_id, group in self.data.groupby('race_id'):
            group = group.sort_values('boat_number')
            if len(group) != 6: continue
            
            res_str = group['sanrentan_result'].iloc[0]
            try:
                actual_result = tuple(map(int, res_str.split('-')))
            except:
                continue
                
            self.races.append({
                'id': race_id,
                'date': group['race_date'].iloc[0],
                'df': group,
                'result': actual_result,
                'payoff': group['sanrentan_payoff'].iloc[0]
            })
        
        self.races.sort(key=lambda x: x['date'])
        split_idx = int(len(self.races) * 0.7) # 7:3
        self.train_races = self.races[:split_idx]
        self.test_races = self.races[split_idx:]
        print(f"Split: Train={len(self.train_races)}, Test={len(self.test_races)}")

    def run_simulation(self, race, weights, n_simulations=50): # シミュレーション数も減らす
        df = race['df']
        ws = weights
        
        potential = (
            df['nat_win_rate'] * ws.get('nat_win_rate', 0) +
            df['loc_win_rate'] * ws.get('loc_win_rate', 0) +
            df['motor_rate'] * ws.get('motor_rate', 0) -
            df['exhibition_time'] * ws.get('exhibition_time', 0)
        )
        
        if df['wind_speed'].iloc[0] > 0.5:
             potential.iloc[0] -= ws.get('wind_speed', 0) * 0.3

        results = []
        for _ in range(n_simulations):
            noise = np.random.normal(0, 0.05, 6)
            total_scores = potential + noise
            rank = np.argsort(total_scores)[::-1] + 1
            results.append(tuple(rank[:3]))
            
        return pd.Series(results).value_counts(normalize=True).to_dict()

    def evaluate(self, races, weights):
        total_spent = 0
        total_return = 0
        bet_count = 0
        
        for race in races:
            probs = self.run_simulation(race, weights)
            for combo, p in probs.items():
                if p >= 0.20: # 確率20%以上に絞る
                    total_spent += 100
                    bet_count += 1
                    if combo == race['result']:
                        total_return += race['payoff']
        
        roi = total_return / total_spent if total_spent > 0 else 0
        return roi, bet_count

    def objective(self, trial):
        weights = {f: trial.suggest_float(f, 0, 10.0) for f in self.features}
        roi, bet_count = self.evaluate(self.train_races, weights)
        
        # 制約：極端に買わないのを防ぐ
        min_bets = len(self.train_races) * 0.3
        if bet_count < min_bets:
            return roi * (bet_count / min_bets)
        return roi

def main():
    backtester = RealDataBacktester(limit_races=100)
    backtester.preprocess()
    
    study = optuna.create_study(direction="maximize")
    print("\nStarting Optimization (Train Set, 15 trials)...")
    study.optimize(backtester.objective, n_trials=15)
    
    best_ws = study.best_params
    print("\n--- Best Weights (Train) ---")
    for k, v in best_ws.items():
        print(f"  {k:20}: {v:.4f}")
        
    print("\n--- Backtesting on Test Set ---")
    test_roi, test_bets = backtester.evaluate(backtester.test_races, best_ws)
    
    print(f"Test ROI: {test_roi*100:.2f}%")
    print(f"Bet Count: {test_bets} (Avg {test_bets/len(backtester.test_races):.2f} per race)")

if __name__ == "__main__":
    main()
