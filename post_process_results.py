import pickle
import os
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def parse_stat_pickle_files(path: Path):
    '''
    path: result directory "8_shadow_models/pubmed-GraphSAGE-online"
    '''
    if not path.exists():
        return
    res_path = path / Path('parsed_results')
    roc_path = res_path / Path('roc_curves')
    Path(roc_path).mkdir(parents=True, exist_ok=True)
    logspace_range = -4
    fpr_space = np.logspace(logspace_range, 0, 1000)
    interpolated_tprs = defaultdict(list)
    table = defaultdict(lambda: defaultdict(list))

    # Parse each stats.pkl file and store values for different audits in a list
    for pickle_file in path.glob('stats_*.pkl'):
        print(pickle_file)
        with open(pickle_file, 'rb') as f:
            stat_dict = pickle.load(f)
        for attack_name, stats in stat_dict.items():
            for fpr, tpr in zip(stats['FPR'], stats['TPR']):
                interp_tpr = np.interp(fpr_space, fpr, tpr)
                interp_tpr[0] = 0.0
                interpolated_tprs[attack_name].append(interp_tpr)
            for quantity, values in stats.items():
                if quantity in ('TPR', 'FPR'):
                    continue
                table[attack_name][quantity].extend(values)

    if not table or not interpolated_tprs:
        raise ValueError(f'Problem with {path}')

    # Save averaged roc curve data and pyplot
    for attack_name, interp_tprs in interpolated_tprs.items():
        interp_tprs = np.stack(interp_tprs, axis=0)
        tpr_mean = np.mean(interp_tprs, axis=0)
        tpr_std = np.std(interp_tprs, axis=0, ddof=1)
        roc_df = pd.DataFrame({
            'fpr': fpr_space,
            'tpr_mean': tpr_mean,
            'tpr_std': tpr_std,
        })
        roc_df.to_csv(f'{roc_path}/roc_{attack_name}_mean.csv', index=False)
        print(f'Saved: {roc_path}/roc_{attack_name}_mean.csv')
        plt.loglog(fpr_space, tpr_mean, label=attack_name)
    plt.loglog(fpr_space, fpr_space, 'k--', label='')
    plt.xlim(10 ** logspace_range, 1)
    plt.ylim(10 ** logspace_range, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(f'{roc_path}/average_roc_curves.png')
    plt.close()

    stat_df = {}
    for attack_name, quantities in table.items():
        row = {}
        for quantity, values in quantities.items():
            values = np.stack(values, axis=0)
            row[f'{quantity}--mean'] = values.mean(0)
            row[f'{quantity}--std'] = values.std(0, ddof=1)
        stat_df[attack_name] = row
    stat_df = pd.DataFrame.from_dict(stat_df, orient='index')
    stat_df.to_csv(f'{res_path}/results.csv')

    quantities = [col.split('--')[0] for col in stat_df.columns]
    percent_metrics = {
        "AUC": "AUC",
        "TPR@0.01FPR": r"TPR@1\%FPR",
        "TPR@0.001FPR": r"TPR@0.1\%FPR",
        "train_acc": "train_acc",
        "test_acc": "test_acc",
    }
    formatted = pd.DataFrame(index=stat_df.index)
    for q in quantities:
        if q in percent_metrics:
            display_name = percent_metrics[q]
            formatted[display_name] = stat_df.apply(
                lambda x: f"${100 * x[f'{q}--mean']:.2f} \\pm {100 * x[f'{q}--std']:.2f}$",
                axis=1,
            )
        elif q == 'time':
            formatted[q] = stat_df.apply(
                lambda x: f"${x[f'{q}--mean']:.5f} \\pm {x[f'{q}--std']:.5f}$", # More precision on time
                axis=1,
            )
        else:
            formatted[q] = stat_df.apply(
                lambda x: f"${x[f'{q}--mean']:.2f} \\pm {x[f'{q}--std']:.2f}$",
                axis=1,
            )

    latex_table = formatted.to_latex(escape=False)  # important: escape=False keeps $ and \pm
    with open(f'{res_path}/latex_table.tex', 'w') as f:
        f.write(latex_table)

def del_parsed_results(path: Path):
    res_path = path / Path('parsed_results')
    shutil.rmtree(res_path, ignore_errors=True)

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    parser = argparse.ArgumentParser()
    parser.add_argument('--resdir', type=str, required=True)
    parser.add_argument('--iterate-subdirs', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    path = Path(args.resdir)
    if args.iterate_subdirs:
        for subfolder in path.iterdir():
            if subfolder.is_dir():
                print(f'Parsing: {subfolder}')
                del_parsed_results(subfolder)
                parse_stat_pickle_files(subfolder)
    else:
        del_parsed_results(path)
        parse_stat_pickle_files(path)
