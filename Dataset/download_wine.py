# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "scikit-learn",
# ]
# ///

import pandas as pd
from sklearn.datasets import load_wine
import os

def main():
    print("Downloading Wine Dataset...")
    
    # 1. Load from sklearn
    wine = load_wine()
    
    # 2. Convert to DataFrame
    df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    
    # 3. Add Target Label
    # 'target' represents the wine class (0, 1, 2)
    df['target'] = wine.target
    
    # 4. Save to CSV
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, 'wine.csv')
    
    df.to_csv(output_path, index=False)
    
    print(f"Success! Wine dataset saved to: {output_path}")
    print(f"Shape: {df.shape} (Rows, Cols)")
    print("Features:", list(wine.feature_names))

if __name__ == "__main__":
    main()
