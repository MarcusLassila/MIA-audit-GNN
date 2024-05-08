import run_mia
import yaml
import pandas as pd
from pathlib import Path

def main():
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    Path('./results').mkdir(parents=True, exist_ok=True)
    stat_frames = []
    static_params = {
        'datadir': './data',
        'savedir': './results',
        'plot_roc': False,
        'experiments': 3,
    }
    for _, params in config.items():
        print()
        print(f'Running MIA experiment.')
        for k, v in params.items():
            print(f'{k}: {v}')
        print()
        params.update(**static_params)
        stat_df, roc_df = run_mia.main(params)
        stat_frames.append(stat_df)
        roc_df.to_csv(f'./results/roc_{params['name']}.csv', index=False)
    pd.concat(stat_frames).to_csv('./results/statistics.csv', sep=',')
    print('Done.')

if __name__ == "__main__":
    main()
