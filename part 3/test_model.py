"""
test_model.py
-------------
Double-click this file (or run it from a terminal) to prove best_model.pkl
works. It loads the model and makes two sample predictions.

Requires: best_model.pkl in the same folder, and the packages in
requirements.txt installed (pip install -r requirements.txt).
"""

import sys

try:
    import joblib
    import pandas as pd
except ImportError as e:
    print("ERROR: a required package is missing ->", e)
    print("Fix: open a terminal in this folder and run:")
    print("    pip install -r requirements.txt")
    input("\nPress Enter to close...")
    sys.exit(1)

try:
    model = joblib.load("best_model.pkl")
    print("Model loaded successfully!")
    print("Model type:", type(model))
except Exception as e:
    print("ERROR: could not load best_model.pkl ->", e)
    print("Make sure best_model.pkl is in the same folder as this script.")
    input("\nPress Enter to close...")
    sys.exit(1)

sample_rows = pd.DataFrame([
    {"age": 45, "bmi": 32.0, "children": 2, "sex_male": 1, "smoker_yes": 1,
     "region_northwest": 0, "region_southeast": 1, "region_southwest": 0},
    {"age": 22, "bmi": 21.5, "children": 0, "sex_male": 0, "smoker_yes": 0,
     "region_northwest": 1, "region_southeast": 0, "region_southwest": 0},
])

predictions = model.predict(sample_rows)
print("\nSample predictions (1 = high cost, 0 = low cost):", predictions)
print("\nIt works! The model file is fine.")

input("\nPress Enter to close...")
