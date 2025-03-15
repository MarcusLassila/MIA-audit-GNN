import mia_audit

import yaml
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import argparse

def main(savedir):
    with open("hyperparam_tuning_config.yaml", "r") as file:
        config = yaml.safe_load(file)
    with open("default_parameters.yaml", "r") as file:
        default_params = yaml.safe_load(file)['default-parameters']
    Path(savedir).mkdir(parents=True, exist_ok=True)
    default_params['savedir'] = savedir
    for audit, params in config.items():
        assert params['hyperparam_search']
        params = default_params | params
        params['name'] = audit
        params['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Running hyperparameter search...')
        print()
        print(yaml.dump(params))
        mia_audit.main(params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir", default="./temp_results", type=str)
    args = parser.parse_args()
    main(args.savedir)
    print('Done.')
