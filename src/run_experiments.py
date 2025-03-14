import run_mia_2

import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse

def main(savedir):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    with open("default_parameters.yaml", "r") as file:
        default_params = yaml.safe_load(file)['default-parameters']
    Path(savedir).mkdir(parents=True, exist_ok=True)
    default_params['savedir'] = savedir
    for experiment, params in config.items():
        params = default_params | params
        params['name'] = experiment
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running MIA...')
        print()
        print(yaml.dump(params))
        stat_df, roc_df = run_mia_2.main(params)
        print(stat_df)
        Path(f'{savedir}/{experiment}').mkdir(parents=True, exist_ok=True)
        stat_df.to_csv(f'{savedir}/{experiment}/results.csv', sep=',')
        roc_df.to_csv(f'{savedir}/{experiment}/roc.csv', sep=',')

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", default="./temp_results", type=str)
    args = parser.parse_args()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    main(args.savedir)
    print('Done.')
