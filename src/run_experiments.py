import run_mia
import yaml

def parse_MIA_output():
    with open("MIA_output.txt", "r") as file:
        pass # TODO: Compute plots comparing auroc, f1, etc. between models and datasets.

def main():
    open("MIA_output.txt", "w").close() # Clear previous outputs before saving new.
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    static_params = {
        'datadir': 'data',
        'savedir': 'plots',
        'experiments': 3,
        'hidden_dim': 256,
    }
    for experiment, params in config.items():
        print(f'Running {experiment} attack.')
        for k, v in params.items():
            print(f'{k}: {v}')
        print()
        params.update(**static_params)
        run_mia.main(params)

if __name__ == "__main__":
    main()
