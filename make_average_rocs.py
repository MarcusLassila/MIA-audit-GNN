import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# Set path to your ROC curve files
data_dir = "/home/johan/project/replicate-MIA-on-GNNs/small_N_results/flickr-GCN/roc_csv"  # <-- change to your folder
csv_files = glob.glob(os.path.join(data_dir, "roc_*_flickr-GCN_*.csv"))

# Define a shared FPR grid (log scale)
mean_fpr = np.logspace(-4, 0, 1000)

# Group files by attack name
attack_files = defaultdict(list)
for file in csv_files:
    # Extract the attack name from the filename
    attack = file.split("flickr-GCN_")[-1].replace(".csv", "")
    attack_files[attack].append(file)

# Prepare output dictionary
attack_results = {}

# Process each attack
for attack, files in attack_files.items():
    interpolated_tprs = []

    for file in files:
        df = pd.read_csv(file)
        if not {"FPR", "TPR"}.issubset(df.columns):
            print(f"Skipping {file}, missing fpr/tpr columns.")
            continue

        fpr = df["FPR"].values
        tpr = df["TPR"].values

        # Interpolate TPR to the common mean_fpr
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interpolated_tprs.append(interp_tpr)

    if not interpolated_tprs:
        continue

    tpr_array = np.stack(interpolated_tprs, axis=0)
    tpr_mean = np.mean(tpr_array, axis=0)
    tpr_std = np.std(tpr_array, axis=0)

    # Save to dictionary
    attack_results[attack] = {
        "fpr": mean_fpr,
        "tpr_mean": tpr_mean,
        "tpr_std": tpr_std,
        "n_runs": len(interpolated_tprs)
    }

    # Optionally: save to CSV
    df_out = pd.DataFrame({
        "fpr": mean_fpr,
        "tpr_mean": tpr_mean,
        "tpr_std": tpr_std
    })
    df_out.to_csv(f"roc_{attack}_mean.csv", index=False)
    print(f"Saved: roc_{attack}_mean.csv")

print("\nDone averaging all attacks.")
