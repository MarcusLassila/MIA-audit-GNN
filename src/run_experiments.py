import run_mia
import yaml
import pandas as pd

OUTPUTFILE = "MIA_output.yaml"

def parse_MIA_output():
    with open(OUTPUTFILE, "r") as file:
        result = yaml.safe_load(file)
        df = pd.DataFrame.from_dict(result).transpose()
        print(df)

def main():
    with open(OUTPUTFILE, "w") as file: # Erase previous content.
        file.write("---\n")
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
        
    static_params = {
        'datadir': './data',
        'savedir': './plots',
        'experiments': 3,
        'outputfile': OUTPUTFILE,
    }
    for _, params in config.items():
        print(f'Running MIA experiment.')
        for k, v in params.items():
            print(f'{k}: {v}')
        print()
        params.update(**static_params)
        run_mia.main(params)
        print()

if __name__ == "__main__":
    main()
    parse_MIA_output()