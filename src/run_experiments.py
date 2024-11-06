import run_mia

import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def add_name(params):
    params['name'] = '-'.join([
        params['attack'],
        params['dataset'],
        params['model'],
    ])

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    Path('./results').mkdir(parents=True, exist_ok=True)
    train_stat_frames = []
    attack_stat_frames = []
    detection_frames = []
    default_params = {
        'datadir': './data',
        'savedir': './results',
        'early_stopping': 0,
        'grid_search': False,
        'optimizer': 'Adam',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'dropout': 0.5,
        'experiments': 10,
        'target_fpr': 0.01,
        'make_roc_plots': True,
        'inductive_split': True,
        'inductive_inference': None,
        'train_frac': 0.5,
        'val_frac': 0.0,
        'seed': 0,
    }
    for _, params in config.items():
        params = default_params | params
        add_name(params)
        print()
        print(f'Running MIA.')
        for k, v in params.items():
            print(f'{k}: {v}')
        print()
        train_stats_df, attack_stats_df, detection_df = run_mia.main(params)
        train_stat_frames.append(train_stats_df)
        attack_stat_frames.append(attack_stats_df)
        detection_frames.append(detection_df)
    pd.concat(train_stat_frames).to_csv(f'{default_params["savedir"]}/train_statistics.csv', sep=',')
    pd.concat(attack_stat_frames).to_csv(f'{default_params["savedir"]}/attack_statistics.csv', sep=',')
    pd.concat(detection_frames).to_csv(f'{default_params["savedir"]}/detection_statistics.csv', sep=',')

if __name__ == "__main__":
    main()
    train_df = pd.read_csv('./results/train_statistics.csv', sep=',')
    attack_df = pd.read_csv('./results/attack_statistics.csv', sep=',')
    detection_df = pd.read_csv('./results/detection_statistics.csv', sep=',')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(train_df)
    print(attack_df)
    print(detection_df)
    print('Done.')
