
import sqlite3
import pandas as pd
import os

DB_PATH = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
OUTPUT_PATH = 'app_data/venue_frame_bias.csv'

def create_bias_table():
    if not os.path.exists(DB_PATH):
        print(f"DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    print("Reading Deme_Ranking from DB...")
    
    query = "SELECT Venue, combination, ratio FROM Deme_Ranking"
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Fetched {len(df)} rows.")
    
    # Process
    # ratio: '6.91%' -> 0.0691 (float)
    df['ratio'] = df['ratio'].str.replace('%', '').astype(float) / 100.0
    
    # Extract 1st Boat from combination '1-2-3'
    # Assume format "X-Y-Z". Boat 1 is X.
    df['boat_number'] = df['combination'].apply(lambda x: int(x.split('-')[0]))
    
    # Standardize Venue Code (ensure string '01', '02', etc.)
    # DB has '01'.
    df['venue_code'] = df['Venue'] # Keep as is
    
    # GroupBy Venue and BoatNumber to get Win Rate
    # Win Rate = Sum of ratios where BoatNumber is 1st
    bias_df = df.groupby(['venue_code', 'boat_number'])['ratio'].sum().reset_index()
    bias_df.rename(columns={'ratio': 'venue_frame_win_rate'}, inplace=True)
    
    print("Bias Table Sample:")
    print(bias_df.head(6))
    
    # Add venue_name for easier mapping if possible? 
    # Or just use venue_code. The app uses venue_name.
    # We should add a map or handle it in app. 
    # train_model.py has `venue_name` but might not have `venue_code`.
    # Actually `inspect_race_data` showed `venue_code` in `race_id`.
    # Let's verify `train_model.py` features.
    # It has `data['venue_code_int']` in feature engineer?
    # Let's save venue_code as is.
    
    if not os.path.exists('app_data'):
        os.makedirs('app_data')
        
    bias_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    create_bias_table()
