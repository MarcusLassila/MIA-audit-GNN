import pickle
import os
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shutil

def parse_stat_pickle_files(prefix, suffices):
    '''
    prefix: result directory + dataset + model, e.g. "8_shadow_models/pubmed-GraphSAGE"
    suffices: list of suffices, e.g. online, offline, classifier etc.
    '''
    fpr_space = np.logspace(-4, 0, 1000)
    for suffix in suffices:
        path = Path(prefix + '-' + suffix)
        res_path = path / Path('parsed_results')
        roc_path = res_path / Path('roc_curves')
        Path(roc_path).mkdir(parents=True, exist_ok=True)
        interpolated_tprs = defaultdict(list)
        table = defaultdict(lambda: defaultdict(list))

        # Parse each stats.pkl file and store values for different audits in a list
        for pickle_file in path.glob('stats_*.pkl'):
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

        # Save averaged roc curve data and pyplot
        for attack_name, interp_tprs in interpolated_tprs.items():
            interp_tprs = np.stack(interp_tprs, axis=0)
            tpr_mean = np.mean(interp_tprs, axis=0)
            tpr_std = np.std(interp_tprs, axis=0)
            roc_df = pd.DataFrame({
                'fpr': fpr_space,
                'tpr_mean': tpr_mean,
                'tpr_std': tpr_std,
            })
            roc_df.to_csv(f'{roc_path}/roc_{attack_name}_mean.csv', index=False)
            print(f'Saved: {roc_path}/roc_{attack_name}_mean.csv')
            plt.loglog(fpr_space, tpr_mean, label=attack_name)
        plt.loglog(fpr_space, fpr_space, 'k--', label='')
        plt.xlim(1e-4, 1)
        plt.ylim(1e-4, 1)
        plt.grid(True)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend()
        plt.savefig(f'{roc_path}/average_roc_curves.png')

        # Save mean+std of other quantities in csv
        stat_df = {}
        for attack_name, quantities in table.items():
            row = {}
            for quantity, values in quantities.items():
                values = np.stack(values, axis=0)
                row[f'{quantity}_mean'] = values.mean(0)
                row[f'{quantity}_std'] = values.std(0)
            stat_df[attack_name] = row
        stat_df = pd.DataFrame.from_dict(stat_df, orient='index')
        stat_df.to_csv(f'{res_path}/results.csv')

def del_parsed_results(prefix, suffices):
    for suffix in suffices:
        path = Path(prefix + '-' + suffix)
        res_path = path / Path('parsed_results')
        shutil.rmtree(res_path, ignore_errors=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resdir', default='./results/8_shadow_models/pubmed-GraphSAGE', type=str)
    parser.add_argument('--suffices', default='online,offline,classifier', type=str)
    args = parser.parse_args()
    res_dir = args.resdir
    suffices = args.suffices.split(',')
    del_parsed_results(res_dir, suffices)
    parse_stat_pickle_files(res_dir, suffices)
