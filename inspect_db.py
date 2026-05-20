import sqlite3
import pandas as pd

conn = sqlite3.connect('boatrace.db')
cursor = conn.cursor()

tables = ['races', 'race_entries', 'results', 'payoffs', 'before_info']
for table in tables:
    print(f"\n--- Table: {table} ---")
    df = pd.read_sql(f"PRAGMA table_info({table})", conn)
    print(df[['name', 'type']])
    print("\nSample Data:")
    try:
        sample = pd.read_sql(f"SELECT * FROM {table} LIMIT 1", conn)
        print(sample)
    except:
        print("Empty or error.")

conn.close()
