
import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys

# Mock Streamlit
class MockSt:
    def markdown(self, x): print(f"[MD] {x}")
    def write(self, x): print(f"[WRITE] {x}")
    def success(self, x): print(f"[SUCCESS] {x}")
    def error(self, x): print(f"[ERROR] {x}")
    def warning(self, x): print(f"[WARNING] {x}")
    def caption(self, x): print(f"[CAPTION] {x}")
    def dataframe(self, x, **kwargs): print(f"[DF] Shape: {x.shape}\n{x.head()}")

sys.modules['streamlit'] = MockSt()
import streamlit as st

# Import App
sys.path.append('d:/BOAT2512_AntiGravity_2_ana')
# Need to prevent app_boatrace from running main on import
# It has `if __name__ == "__main__":`?
# No, `app_boatrace.py` runs top-level code (Streamlit style).
# So importing it might run it.
# I should just replicate the logic in this script using the same constants.

MODEL_PATH = 'model_honmei.txt'

def verify():
    print("Loading Model...")
    model = lgb.Booster(model_file=MODEL_PATH)
    
    # Create Dummy Data (6 boats) with specific scores to test Std
    # Case 1: High Std (Enjoy) -> [10, 8, 2, 1, 0, 0] -> Std approx 4
    # Case 2: Low Std (Chaos) -> [5, 5, 5, 5, 5, 5] -> Std 0
    # Case 3: Mid Std (Skip) -> [6, 5, 4, 3, 2, 1] -> Std approx 1.8
    
    cases = [
        {'name': 'High Std (Enjoy)', 'scores': [10.0, 8.0, 2.0, 1.0, 0.0, 0.0]},
        {'name': 'Low Std (Chaos)', 'scores': [5.1, 5.0, 4.9, 5.2, 4.8, 5.0]},
        {'name': 'Mid Std (Skip)', 'scores': [6.0, 5.0, 4.0, 3.5, 3.0, 2.0]}
    ]
    
    TH_HIGH = 1.5347
    TH_LOW = 1.2923
    
    for c in cases:
        print(f"\n--- Testing {c['name']} ---")
        scores = np.array(c['scores'])
        std = np.std(scores, ddof=1) # Pandas uses ddof=1 by default
        print(f"Std: {std:.4f}")
        
        mode = "Skip"
        if std >= TH_HIGH: mode = "Enjoy"
        elif std <= TH_LOW: mode = "Chaos"
        
        print(f"Mode: {mode}")
        
        if mode == "Enjoy":
            print("Action: Top 4, No Filter")
        elif mode == "Chaos":
            print("Action: Top 6, Odds >= 30.0")
        else:
            print("Action: Skip")
            
        # Verify Thresholds correct
        if c['name'] == 'High Std (Enjoy)' and mode != 'Enjoy': print("FAIL")
        elif c['name'] == 'Low Std (Chaos)' and mode != 'Chaos': print("FAIL")
        elif c['name'] == 'Mid Std (Skip)' and mode != 'Skip': print("FAIL (Expected Skip)")
        else: print("PASS")

if __name__ == "__main__":
    verify()
