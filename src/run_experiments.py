import run_mia
import utils
import yaml
import pandas as pd
from pathlib import Path

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    Path('./results').mkdir(parents=True, exist_ok=True)
    stat_frames = []
    roc_frames = []
    static_params = {
        'batch_size': 32,
        'datadir': './data',
        'savedir': './results',
        'plot_roc': True,
        'early_stopping': True,
        'experiments': 3,
    }
    for _, params in config.items():
        print()
        print(f'Running MIA.')
        for k, v in params.items():
            print(f'{k}: {v}')
        print()
        params.update(**static_params)
        stat_df, roc_df = run_mia.main(params)
        stat_frames.append(stat_df)
        roc_frames.append(roc_df)
    pd.concat(stat_frames).to_csv(f'{static_params["savedir"]}/statistics.csv', sep=',')
    pd.concat(roc_frames, axis=1).to_csv(f'{static_params["savedir"]}/rocs.csv', sep=',', index=False)
    print('Done.')

if __name__ == "__main__":
    main()
    utils.plot_roc_csv("./results/rocs.csv")