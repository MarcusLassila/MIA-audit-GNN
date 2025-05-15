import os
import pandas as pd
import sys
import re
from collections import defaultdict

ATTACK_DICT = {
    'MLP-attack-0hop': 'MLP (0-hop)',
    'MLP-attack-comb': 'MLP (0+2-hop)',
    'lira': 'LiRA',
    'lira-offline': 'LiRA (off)',
    'rmia': 'RMIA',
    'rmia-offline': 'RMIA (off)',
    'lset': '\\textbf{BASE}',
    'lset-offline': '\\textbf{BASE} (off)',
    'graph-lset-MI': '\\textbf{G-BASE}',
    'graph-lset-MIA': '\\textbf{G-BASE}',
    'graph-lset-MI-offline': '\\textbf{G-BASE} (off)',
    'graph-lset-MIA-offline': '\\textbf{G-BASE} (off)',
    'graph-lset-MI-offline-0.9': '\\textbf{G-BASE} (off)',
    'graph-lset-MI-offline-no-scale': '\\textbf{G-BASE} (off) nohyp',
}

def attack_map(attack):
    return ATTACK_DICT[attack]

def convert_to_latex_format(s: str, make_bold: bool):
    mean, std = s.replace('(', '').replace(')', '').split()
    mean = float(mean) * 100
    std = float(std) * 100
    if make_bold:
        return "$\\mathbf{" + f"{mean:.2f} \\pm {std:.2f}" + "}$"
    else:
        return f"${mean:.2f} \\pm {std:.2f}$"

def extract_first_number(s):
    try:
        return float(s.split()[0])
    except (AttributeError, IndexError, ValueError):
        return float('nan')

def max_mean_value_indices(df, columns):
    res = {}
    for col in columns:
        if col == 'Unnamed: 0':
            continue
        numeric_series = df[col].map(extract_first_number)
        res[col] = numeric_series.idxmax()
    return res

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
                        offline_mask = df["Unnamed: 0"].str.contains("offline", case=False, na=False)
                        df_online = df[~offline_mask].copy()
                        df_offline = df[offline_mask].copy()
                        for df_i in df_online, df_offline:
                            idx_for_bold = max_mean_value_indices(df_i, selected_columns)
                            for idx, row in df_i.iterrows():
                                attack = row['Unnamed: 0'].split('_')[1]
                                try:
                                    attack = attack_map(attack)
                                except KeyError:
                                    continue
                                metrics = " & "
                                cols = []
                                for col in selected_columns:
                                    if col == 'Unnamed: 0':
                                        continue
                                    best = idx_for_bold[col]
                                    make_bold = best == idx
                                    cols.append(convert_to_latex_format(row[col], make_bold=make_bold))
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
