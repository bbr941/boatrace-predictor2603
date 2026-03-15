
import sqlite3

def list_tables():
    # Use absolute path found in make_data_set.py
    db_path = r'D:\BOAT2504_Base_line\BOAT2504_DB\boatrace.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # List tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables:", [t[0] for t in tables])
    
    # Check if 'Deme_Ranking' exists and show schema
    if ('Deme_Ranking',) in tables or ('Deme_Ranking',) in [(t[0],) for t in tables]:
        print("\nSchema of 'Deme_Ranking':")
        cursor.execute("PRAGMA table_info(Deme_Ranking)")
        schema = cursor.fetchall()
        for col in schema:
            print(col)
            
        print("\nFirst 5 rows of 'Deme_Ranking':")
        cursor.execute("SELECT * FROM Deme_Ranking LIMIT 5")
        rows = cursor.fetchall()
        for r in rows:
            print(r)
    else:
        # Check race_results?
        print("\n'results' table not found. Checking other potential names...")
        
    conn.close()

if __name__ == "__main__":
    list_tables()
