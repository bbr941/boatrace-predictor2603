import pandas as pd
import numpy as np
import itertools

def run_monte_carlo_simulation(num_simulations=10000, threshold=1.1):
    """
    ボートレースのモンテカルロ・シミュレーション プロトタイプ
    """
    
    # 1. ダミーデータの設定
    # 各艇の平均ST、標準偏差、コース別勝率（簡易的に1着率、2着率、3着率を定義）
    data = {
        'boat': [1, 2, 3, 4, 5, 6],
        'avg_st': [0.12, 0.15, 0.16, 0.14, 0.17, 0.18],  # 平均スタートタイミング
        'std_st': [0.03, 0.04, 0.04, 0.03, 0.05, 0.06],  # STの標準偏差
        'win_rate_1st': [0.55, 0.15, 0.10, 0.10, 0.05, 0.05], # 1着率のベース
        'win_rate_2nd': [0.20, 0.30, 0.20, 0.15, 0.10, 0.05], # 2着率のベース
        'win_rate_3rd': [0.10, 0.20, 0.25, 0.20, 0.15, 0.10], # 3着率のベース
    }
    df_boats = pd.DataFrame(data)
    
    # 2. スタートシミュレーション (正規分布)
    # 1万回分の各艇のSTを一括生成
    st_sims = np.random.normal(
        df_boats['avg_st'].values, 
        df_boats['std_st'].values, 
        (num_simulations, 6)
    )
    # フライング(F)対策など下限を設ける（簡易的に0.01秒以下は0.01とする）
    st_sims = np.maximum(st_sims, 0.01)

    # 各シミュレーションでの着順を格納するリスト
    results = []

    # 3. 展開ロジック（簡易版）
    for i in range(num_simulations):
        sts = st_sims[i]
        
        # 基本スコア（勝率ベースの期待値にノイズを加える）
        # スコアが高いほど着順が上になると定義
        base_scores = (
            df_boats['win_rate_1st'] * 3 + 
            df_boats['win_rate_2nd'] * 2 + 
            df_boats['win_rate_3rd'] * 1
        ).values
        
        # スタート優劣によるスコア変動
        # STが早い(数値が小さい)ほど加点。平均より早い場合にブースト。
        st_bonus = (0.2 - sts) * 10  # 0.2秒を基準に早いほど大きく加点
        
        # 簡易展開条件：4カド捲り
        # 4号艇が3号艇より0.05秒以上早い場合、4号艇にスリットブースト
        if sts[3] < sts[2] - 0.05:
            st_bonus[3] += 5.0
            # 1-3号艇は捲られるリスクで減点
            st_bonus[0:3] -= 2.0
            
        # 最終スコア算出（基本能力 + スタート効果 + 運要素のノイズ）
        noise = np.random.normal(0, 1.0, 6)
        total_scores = base_scores + st_bonus + noise
        
        # スコア順に艇番を並び替え
        rankings = np.argsort(total_scores)[::-1] + 1 # インデックスは0-5なので+1
        results.append(tuple(rankings[:3]))

    # 4. 期待値と閾値の算出
    # 結果の集計
    series_results = pd.Series(results)
    prob_df = series_results.value_counts(normalize=True).reset_index()
    prob_df.columns = ['combo', 'prob']
    
    # 120通りの全組み合わせを生成
    all_combos = list(itertools.permutations([1, 2, 3, 4, 5, 6], 3))
    df_all = pd.DataFrame({'combo': all_combos})
    
    # シミュレーション結果と結合
    df_final = pd.merge(df_all, prob_df, on='combo', how='left').fillna(0)
    
    # 仮想オッズの設定（ダミー値：シミュレーション確率の逆数にマージンを加えたものにランダム性を付与）
    # 実際は外部データから取得する部分。デモ用に期待値がばらけるように調整。
    df_final['dummy_odds'] = (1.0 / (df_final['prob'] + 0.002)) * np.random.uniform(0.5, 2.0, len(df_final))
    df_final['dummy_odds'] = df_final['dummy_odds'].clip(lower=1.0, upper=500.0)
    
    # 期待値計算
    df_final['expected_value'] = df_final['prob'] * df_final['dummy_odds']
    
    # 閾値によるフィルタリング
    bet_list = df_final[df_final['expected_value'] >= threshold].sort_values('expected_value', ascending=False)
    
    # 出力
    print(f"--- Simulation Summary ({num_simulations} times) ---")
    print(f"Threshold: {threshold}")
    print(f"Recommended Bets: {len(bet_list)}")
    print(f"Estimated Win Probability (Total): {bet_list['prob'].sum():.2%}")
    print("\nTop 10 Recommended Bets:")
    print(bet_list.head(10)[['combo', 'prob', 'dummy_odds', 'expected_value']].to_string(index=False))
    
    return df_final, bet_list

if __name__ == "__main__":
    # 期待値1.1以上を抽出
    df, bets = run_monte_carlo_simulation(num_simulations=10000, threshold=1.1)
    
    print("\n--- Additional Threshold Checks ---")
    for t in [1.0, 1.2, 1.5]:
        count = len(df[df['expected_value'] >= t])
        print(f"Threshold {t:.1f}: {count} bets")
