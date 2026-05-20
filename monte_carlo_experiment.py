import pandas as pd
import numpy as np
import itertools
import optuna
from tqdm import tqdm

class MonteCarloExplorer:
    def __init__(self, n_races=20):
        self.n_races = n_races
        self.features = [
            'national_win_rate', 'local_win_rate', 'motor_rate', 
            'exhibition_time', 'avg_st', 'wind_speed'
        ]
        self.races_data = self.generate_dummy_races(n_races)

    def generate_dummy_races(self, n_races):
        """
        複数レース分の多角的なダミーデータを生成
        """
        races = []
        for r in range(n_races):
            # 艇ごとの特徴量
            race_df = pd.DataFrame({
                'boat': range(1, 7),
                'national_win_rate': np.random.uniform(3.0, 8.0, 6),
                'local_win_rate': np.random.uniform(3.0, 8.0, 6),
                'motor_rate': np.random.uniform(20.0, 50.0, 6),
                'exhibition_time': np.random.uniform(6.5, 6.9, 6),
                'avg_st': np.random.uniform(0.12, 0.20, 6),
            })
            # 1号艇を少し強く設定（ベースライン）
            race_df.loc[0, 'national_win_rate'] += 1.0
            
            # 外部環境（レース全体で共通）
            wind_speed = np.random.uniform(0, 8)
            
            # 正解データ（Ground Truth）の生成用：特定のルールで勝敗を決める
            # 例: national_win_rateが高いほど勝ちやすい + 運
            true_scores = race_df['national_win_rate'] * 2.0 + np.random.normal(0, 1.0, 6)
            true_rank = np.argsort(true_scores)[::-1] + 1
            result_3rentan = tuple(true_rank[:3])
            
            # 仮想オッズ（正解の確率に基づいて生成、少し歪ませる）
            all_combos = list(itertools.permutations(range(1, 7), 3))
            odds_df = pd.DataFrame({'combo': all_combos})
            # 1-2-3 など人気どころは低めに、大穴は高めに
            odds_df['odds'] = np.random.uniform(10, 500, len(all_combos))
            
            races.append({
                'id': r,
                'df': race_df,
                'wind_speed': wind_speed,
                'result': result_3rentan,
                'odds': odds_df.set_index('combo')['odds'].to_dict()
            })
        return races

    def run_simulation(self, race_data, weights, n_simulations=500):
        """
        パラメータ化された重み付きシミュレーション
        """
        df = race_data['df']
        ws = weights
        
        # 特徴量の正規化と重み付け
        # 勝率系(高い方がいい)とタイム/ST系(低い方がいい)を分ける
        potential = (
            df['national_win_rate'] * ws.get('national_win_rate', 0) +
            df['local_win_rate'] * ws.get('local_win_rate', 0) +
            df['motor_rate'] * ws.get('motor_rate', 0) -
            df['exhibition_time'] * 10 * ws.get('exhibition_time', 0) - # タイムは引く
            df['avg_st'] * 20 * ws.get('avg_st', 0)
        )
        
        # 風速の影響：風が強いとインが弱くなる等の簡易ロジック
        if race_data['wind_speed'] > 5:
            potential.iloc[0] -= ws.get('wind_speed', 0) * 2.0

        results = []
        for _ in range(n_simulations):
            # ノイズを加えて順位決定
            noise = np.random.normal(0, 3.0, 6)
            total_scores = potential + noise
            rank = np.argsort(total_scores)[::-1] + 1
            results.append(tuple(rank[:3]))
            
        return pd.Series(results).value_counts(normalize=True).to_dict()

    def evaluate(self, weights, threshold=1.2):
        """
        全レースでシミュレーションを実行し、ROIと的中頻度を算出
        """
        total_spent = 0
        total_return = 0
        bet_count = 0
        
        for race in self.races_data:
            probs = self.run_simulation(race, weights)
            odds = race['odds']
            
            # 期待値計算
            for combo, p in probs.items():
                ev = p * odds.get(combo, 1.0)
                if ev >= threshold:
                    # ベット実行
                    total_spent += 100
                    bet_count += 1
                    if combo == race['result']:
                        total_return += 100 * odds[combo]
        
        roi = total_return / total_spent if total_spent > 0 else 0
        bet_frequency = bet_count / (self.n_races * 120) # 3連単全通りベース
        # レース単位での的中頻度（少なくとも1点購入したレースの割合）
        race_bet_count = 1 if total_spent > 0 else 0 # 簡略化
        
        return roi, bet_count

    def objective(self, trial):
        weights = {
            f: trial.suggest_float(f, 0.0, 5.0) for f in self.features
        }
        
        roi, bet_count = self.evaluate(weights)
        
        # 制約条件: ベット機会が少なすぎる（例: 5点以下）場合はペナルティ
        min_bets = self.n_races * 0.5 # 20レースで合計10点以上は買ってほしい
        if bet_count < min_bets:
            # ROIを大幅に下げる
            return roi * (bet_count / min_bets)
        
        return roi

def run_experiment():
    print("Initializing Experiment Framework...")
    explorer = MonteCarloExplorer(n_races=30)
    
    print("Starting Optimization with Optuna (30 trials)...")
    study = optuna.create_study(direction="maximize")
    study.optimize(explorer.objective, n_trials=30)
    
    print("\n--- Optimization Results ---")
    print(f"Best ROI Score: {study.best_value:.4f}")
    print("Best Weights (Feature Importance):")
    for feat, val in study.best_params.items():
        importance = "High" if val > 3.0 else ("Medium" if val > 1.0 else "Low/Noise")
        print(f"  {feat:20}: {val:.4f} ({importance})")
        
    return study.best_params

if __name__ == "__main__":
    run_experiment()
