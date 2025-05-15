import pickle
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ATTACK_DICT = {
    'MLP-attack-0hop': 'MLP-classifier (0-hop)',
    'MLP-attack-comb': 'MLP-classifier (0+2-hop)',
    'lira': 'LiRA',
    # 'lira-offline': 'LiRA (offline)',
    'rmia': 'RMIA',
    # 'rmia-offline': 'RMIA (offline)',
    'lset': 'BASE (ours)',
    # 'lset-offline': 'BASE (offline)',
    'graph-lset-MI': 'G-BASE (ours)',
    'graph-lset-MIA': 'G-BASE (ours)',
    # 'graph-lset-MI-offline': 'G-BASE (offline)',
    # 'graph-lset-MIA-offline': 'G-BASE (offline)',
}

ORDER = ['BASE (ours)', 'G-BASE (ours)', 'LiRA', "RMIA", "MLP-classifier (0-hop)", "MLP-classifier (0+2-hop)", "Random guess"]

def attack_map(attack):
    return ATTACK_DICT[attack]

def attack_filter(attack):
    return attack in set(ATTACK_DICT.keys())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="./", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--make-plots", action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    path = Path(args.root + '/' + args.name)
    name = args.name
    stats_list = []
    for pickle_file in path.glob('stats_*.pkl'):
        with open(pickle_file, 'rb') as f:
            stats = pickle.load(f)
            stats_list.append(stats)
    frames = []
    attacks = list(stats_list[0].keys())
    n_audits = None
    for attack in attacks:
        table = defaultdict(list)
        for key in stats_list[0][attack].keys():
            if key in ('TPR', 'FPR'):
                continue
            concat_value = []
            for stats in stats_list:
                value = stats[attack][key]
                if n_audits is None:
                    n_audits = len(value)
                else:
                    assert n_audits == len(value), "All index must contain same number of audits"
                concat_value.append(value)
            concat_value = np.concatenate(concat_value)
            stat = f'{concat_value.mean():.4f} ({concat_value.std():.4f})'
            table[key].append(stat)
        frames.append(pd.DataFrame(table, index=[name + '_' + attack]))
    stat_df = pd.concat(frames)
    stat_df.to_csv(f'{path}/results.csv', sep=',')
    if args.make_plots:
        running_index = 0
        for stats in stats_list:
            for i in range(n_audits):
                lines = []
                plt.clf()
                plt.figure(figsize=(5.5, 5.5))
                for attack in filter(attack_filter, attacks):
                    fpr = stats[attack]['FPR'][i]
                    tpr = stats[attack]['TPR'][i]
                    label = attack_map(attack)
                    line, = plt.loglog(fpr, tpr, label=label)
                    lines.append(line)
                x, y = [0, 1], [0, 1]
                line, = plt.loglog(x, y, linestyle='--', color='black', label='Random guess')
                lines.append(line)
                lines.sort(key=lambda line: ORDER.index(line.get_label()))
                plt.legend(handles=lines)
                plt.xlim(1e-4, 1)
                plt.ylim(1e-4, 1)
                plt.grid(True)
                plt.xlabel('False Positive Rate', fontsize=13)
                plt.ylabel('True Positive Rate', fontsize=13)
                if args.root == 'large_N_results':
                    num_shadow_models = '128'
                else:
                    num_shadow_models = '8'
                plt.savefig(f'{path}/{name}_{num_shadow_models}_roc_{running_index}.png')
                running_index += 1
