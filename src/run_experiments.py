import hypertuner
import run_mia

import yaml
import pandas as pd
from pathlib import Path

def add_name(params):
    params['name'] = '-'.join([params['attack'], params['dataset'], params['model'], f"{params['query_hops']}hop"])

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    Path('./results').mkdir(parents=True, exist_ok=True)
    stat_frames = []
    roc_frames = []
    static_params = {
        'datadir': './data',
        'savedir': './results',
        'early_stopping': True,
        'grid_search': False,
        'optimizer': 'Adam',
        'lr': 0.01,
        'weight_decay': 1e-4,
        'dropout': 0.5,
        'experiments': 5,
        'make_plots': True,
        'hidden_dim_target': 32,
        'query_hops': 0,
    }
    for _, params in config.items():
        params.update(**static_params)
        add_name(params)
        print()
        print(f'Running MIA.')
        for k, v in params.items():
            print(f'{k}: {v}')
        print()
        stat_df, roc_df = run_mia.main(params)
        stat_frames.append(stat_df)
        roc_frames.append(roc_df)
    pd.concat(stat_frames).to_csv(f'{static_params["savedir"]}/statistics.csv', sep=',')
    pd.concat(roc_frames, axis=1).to_csv(f'{static_params["savedir"]}/rocs.csv', sep=',', index=False)

if __name__ == "__main__":
    main()
    df = pd.read_csv('./results/statistics.csv', sep=',').set_index('Unnamed: 0')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)
    print('Done.')
