import pandas as pd
import os
import re
import argparse
from pathlib import Path

                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--filepath", type=str)
    args = parser.parse_args()

    path = os.path.dirname(args.filepath)
    Path(path + '/roc_csv').mkdir(parents=True, exist_ok=True)
    # Load your original CSV
    df = pd.read_csv(args.filepath)  # Replace with your actual file name

    # Loop through columns to find FPR-TPR pairs
    for col in df.columns:
        if col.startswith("FPR_"):
            # Extract the identifier from the column name
            match = re.match(r"FPR_(\d+)_(.+)", col)
            if match:
                n, id_ = match.groups()
                tpr_col = f"TPR_{n}_{id_}"
                if tpr_col in df.columns:
                    # Extract the two columns
                    roc_df = df[[col, tpr_col]].copy()
                    roc_df.columns = ["FPR", "TPR"]  # Rename for consistency
                    roc_df = roc_df.dropna(how='any')  # Drop rows with any NaNs
                    roc_df = roc_df[(roc_df['FPR'].astype(str).str.strip() != '') & 
                                    (roc_df['TPR'].astype(str).str.strip() != '')]
                    # Save to new CSV
                    filename = f"{path}/roc_csv/roc_{n}_{id_}.csv"
                    roc_df.to_csv(filename, index=False)
                    print(f"Saved {filename}")
                else:
                    print(f"Warning: TPR column {tpr_col} not found for {col}")