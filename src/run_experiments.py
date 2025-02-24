import run_mia_2

import yaml
import numpy as np
import pandas as pd
import torch
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
    with open("default_parameters.yaml", "r") as file:
        default_params = yaml.safe_load(file)['default-parameters']
    Path('./results').mkdir(parents=True, exist_ok=True)
    stat_frames = []
    for _, params in config.items():
        params = default_params | params
        add_name(params)
        print()
        print(f'Running MIA.')
        for k, v in params.items():
            print(f'{k}: {v}')
        print()
        stats_df = run_mia_2.main(params)
        stat_frames.append(stats_df)
    pd.concat(stat_frames).to_csv(f'{default_params["savedir"]}/results.csv', sep=',')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
    stat_df = pd.read_csv('./results/results.csv', sep=',')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(stat_df)
    print('Done.')
