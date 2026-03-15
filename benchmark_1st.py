
import pandas as pd
import train_model

DATA_PATH = 'boatrace_dataset_labeled_v2.csv'

def benchmark_boat1():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # Split (Same logic as train)
    unique_races = df['race_id'].unique()
    split_idx = int(len(unique_races) * 0.8)
    test_races = unique_races[split_idx:]
    test_df = df[df['race_id'].isin(test_races)].copy()
    
    print(f"Test Set: {len(test_races)} races")
    
    # Strategy: Always bet on Boat 1
    # Filter for Boat 1
    boat1_df = test_df[test_df['boat_number'] == 1].copy()
    
    # Check if Boat 1 won
    wins = boat1_df[boat1_df['rank'] == 1]
    hits = len(wins)
    
    hit_rate = hits / len(test_races)
    
    # Recovery
    # Assume we bet 100 yen on every race on Boat 1
    total_bet = len(test_races) * 100
    
    # Return
    # We need odds.
    has_odds = 'odds' in boat1_df.columns
    if has_odds:
        total_return = wins['odds'].sum() * 100
    else:
        # Proxy
        total_return = (1 / wins['syn_win_rate']).sum() * 100
        
    recovery_rate = total_return / total_bet
    
    print("\n=== Benchmark: Always Bet Boat 1 ===")
    print(f"Hit Rate: {hit_rate:.2%}")
    print(f"Recovery Rate: {recovery_rate:.2%}")

if __name__ == "__main__":
    benchmark_boat1()
