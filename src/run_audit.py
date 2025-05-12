import mia_audit

import yaml
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse

def main(savedir, index, seed):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    with open("default_parameters.yaml", "r") as file:
        default_params = yaml.safe_load(file)['default-parameters']
    Path(savedir).mkdir(parents=True, exist_ok=True)
    default_params['savedir'] = savedir
    for audit, params in config.items():
        params = default_params | params
        assert not params['hyperparam_search']
        params['name'] = audit
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        if seed != -1:
            print(f'Overwriting config seed with seed={seed}')
            params['seed'] = seed
        print('Running MIA...')
        print()
        print(yaml.dump(params))
        stat_df, stats = mia_audit.run(params)
        print(stat_df)
        Path(f'{savedir}/{audit}').mkdir(parents=True, exist_ok=True)
        if index == 0:
            stat_df.to_csv(f'{savedir}/{audit}/results.csv', sep=',')
        else:
            stat_df.to_csv(f'{savedir}/{audit}/results_{index}.csv', sep=',')
        with open(f'{savedir}/{audit}/stats_{index}.pkl', 'wb') as f:
            pickle.dump(stats, f)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", default="./temp_results", type=str)
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--seed", default=-1, type=int)
    args = parser.parse_args()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    main(args.savedir, args.index, args.seed)
    print('Done.')
