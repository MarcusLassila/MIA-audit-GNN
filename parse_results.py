import os
import pandas as pd
import sys
import re
from collections import defaultdict

ATTACK_DICT = {
    'MLP-attack-0hop': 'MLP (0-hop)',
    'MLP-attack-comb': 'MLP (0+2-hop)',
    'lira': 'LiRA',
    'lira-offline': 'LiRA (offline)',
    'rmia': 'RMIA',
    'rmia-offline': 'RMIA (offline)',
    'lset': 'LSET',
    'lset-offline': 'LSET (offline)',
    'graph-lset-MI': 'GraphLSET',
    'graph-lset-MIA': 'GraphLSET (MIA)',
    'graph-lset-MI-offline': 'GraphLSET (offline)',
    'graph-lset-MI-offline-0.9': 'GraphLSET (offline)',
    'graph-lset-MI-offline-no-scale': 'GraphLSET (offline) nohyp',
}

def attack_map(attack):
    return ATTACK_DICT[attack]

def parse_csv_files(base_path, directories, selected_columns, res_dict):
    for directory in directories:
        full_path = os.path.join(base_path, directory)
        if os.path.isdir(full_path):
            print(f"Searching in: {full_path}")
            for file in os.listdir(full_path):
                if file == 'results.csv':
                    file_path = os.path.join(full_path, file)
                    with open(file_path, newline='') as f:
                        df = pd.read_csv(f, usecols=selected_columns)
                        for _, row in df.iterrows():
                            attack = row['Unnamed: 0'].split('_')[1]
                            try:
                                attack = attack_map(attack)
                            except KeyError:
                                continue
                            metrics = " & "
                            value_pattern = r"(\d+(?:\.\d+)?)\s*\((\d+(?:\.\d+)?)\)"
                            repl = r"$\1 \\pm \2$ "
                            cols = []
                            for col in selected_columns:
                                if col == 'Unnamed: 0':
                                    continue
                                cols.append(re.sub(value_pattern, repl, row[col]))
                            metrics += '& '.join(cols)
                            res_dict[attack] += metrics
        else:
            print(f"Directory not found: {full_path}")

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python parse_result.py resdir"
    root = sys.argv[1]
    dir_groups = [
        'cora-GCN,citeseer-GAT,pubmed-GraphSAGE'.split(','),
        'flickr-GCN,amazon-photo-GAT,github-GraphSAGE'.split(','),
    ]
    selected_columns = 'Unnamed: 0,AUC,TPR@0.01FPR,TPR@0.001FPR'.split(',')
    with open(f'{root}/latex_table.tex', 'w') as f:
        f.write('')
    for directories in dir_groups:
        res_dict = defaultdict(str)
        parse_csv_files(root, directories, selected_columns, res_dict)
        table_entry = ''
        for attack, metrics in res_dict.items():
            table_entry += '& ' + attack + ' ' + metrics + '\\\\\n'
        with open(f'{root}/latex_table.tex', 'a') as f:
            f.write(table_entry)
            f.write('\n')
