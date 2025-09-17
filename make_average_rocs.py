import os
import glob
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path

def average_rocs(data_dir, log_range=-4):
    '''
    data_dir: string containing the path to the roc csv files of TPR/FPR data
    log_range: compute roc curve in log scales from log_range to 0
    '''
    dataset_model = data_dir.split("/")[-2]
    csv_files = glob.glob(os.path.join(data_dir, f"roc_*{dataset_model}_*.csv"))

    # Define a shared FPR grid (log scale)
    mean_fpr = np.logspace(log_range, 0, 1000)

    # Group files by attack name
    attack_files = defaultdict(list)
    for file in csv_files:
        attack = file.split(f"{dataset_model}_")[-1].replace(".csv", "")
        attack_files[attack].append(file)

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

        df_out = pd.DataFrame({
            "fpr": mean_fpr,
            "tpr_mean": tpr_mean,
            "tpr_std": tpr_std
        })
        savedir = f"{data_dir}/average_rocs"
        Path(f"{data_dir}/average_rocs").mkdir(parents=False, exist_ok=True)
        df_out.to_csv(f"{savedir}/roc_{attack}_mean.csv", index=False)
        print(f"Saved: {savedir}/roc_{attack}_mean.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Path to roc.csv directory.")
    parser.add_argument("datadir", type=str, help="Path to the roc.csv directory")
    args = parser.parse_args()
    if os.path.isdir(args.datadir):
        print(f"Directory exists: {args.datadir}")
    else:
        print(f"Invalid directory: {args.datadir}")
    average_rocs(args.datadir, log_range=-4)
