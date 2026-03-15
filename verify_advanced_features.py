
import pandas as pd
import sys
import os

# App Dir
sys.path.append('d:/BOAT2512_AntiGravity_2_ana')
from app_boatrace import BoatRaceScraper, FeatureEngineer

def verify():
    # Test Race: Kiryu 12R (Already known)
    # 20251227, 01, 12
    print("Fetching race data...")
    df = BoatRaceScraper.get_race_data("20251227", "01", 12)
    if df is None:
        print("Failed to scrape.")
        return

    print("Processing features...")
    # This calls add_advanced_features internally
    df_processed = FeatureEngineer.process(df, "桐生", debug_mode=True)
    
    # Check new columns
    new_cols = [
        'is_F_holder', 'corrected_st', 'inner_st_gap_corrected',
        'motor_gap', 'specialist_score', 
        'sashi_potential', 'makuri_potential', 'venue_frame_win_rate'
    ]
    
    print("\n--- Advanced Features Verification ---")
    if all(c in df_processed.columns for c in new_cols):
        print("SUCCESS: All new columns present.")
    else:
        missing = [c for c in new_cols if c not in df_processed.columns]
        print(f"FAILURE: Missing columns: {missing}")
        
    print("\nSample Data (First 3 rows):")
    print(df_processed[['boat_number', 'is_F_holder', 'corrected_st', 'motor_rank', 'tenji_rank', 'motor_gap', 'venue_frame_win_rate']].head(6))
    
    # Check logic
    # Boat 1 was Absent in this race (filtered out by valid logic).
    # Boat 2: Check motor_gap.
    # Check venue_frame_win_rate for Kiryu (01) Boat 1 (if present) or others.
    
if __name__ == "__main__":
    verify()
