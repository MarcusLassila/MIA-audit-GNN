import os
import pandas as pd
import sys
import re
from collections import defaultdict


def parse_csv_files(model_architecture, base_path, datasets, selected_columns, res_dict):
    for dataset in datasets:
        dir_name = os.path.join(base_path, dataset, f"{dataset}-{model_architecture}")
        if os.path.isdir(dir_name):
            print(f"Searching in: {dir_name}")
            for file in os.listdir(dir_name):
                if file == 'results.csv':
                    file_path = os.path.join(dir_name, file)
                    with open(file_path, newline='') as f:
                        df = pd.read_csv(f, usecols=selected_columns)
                        for _, row in df.iterrows():
                            attack_pattern = r"^[a-zA-Z0-9-]+-[a-zA-Z0-9]+_([a-zA-Z0-9-]+)"
                            m = re.match(attack_pattern, row['Unnamed: 0'])
                            if not m:
                                print('no match:')
                                print(row['Unnamed: 0'])
                                continue
                            key = m.group(1)
                            value = "& "
                            value_pattern = r"(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\)"
                            repl = r"$\1 \\pm \2$ "
                            auc = re.sub(value_pattern, repl, row.AUC)
                            fpr_1 = re.sub(value_pattern, repl, row['TPR@0.01FPR'])
                            fpr_2 = re.sub(value_pattern, repl, row['TPR@0.001FPR'])
                            value += '& '.join([auc, fpr_1, fpr_2])
                            res_dict[key] += value
        else:
            print(f"Directory not found: {dir_name}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python parse_result.py resdir model")
    else:
        root = sys.argv[1]
        letter = sys.argv[2]
        dataset_groups = ['cora,citeseer,pubmed'.split(','), 'amazon-photo,flickr'.split(',')]
        selected_columns = 'Unnamed: 0,AUC,TPR@0.01FPR,TPR@0.001FPR'.split(',')
        for datasets in dataset_groups:
            res_dict = defaultdict(str)
            parse_csv_files(letter, root, res_dict)
            table_entry = ''
            for key, value in res_dict.items():
                table_entry += '& ' + key + ' ' + value + '\\\\\n'
            with open(f'{root}/latex_table.tex', 'w') as f:
                f.write(table_entry)
