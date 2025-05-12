import pickle
from collections import defaultdict
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    stat_df.to_csv(f'{path}/recombined_results.csv', sep=',')
    if args.make_plots:
        running_index = 0
        for stats in stats_list:
            for i in range(n_audits):
                legend = []
                plt.clf()
                for attack in attacks:
                    fpr = stats[attack]['FPR'][i]
                    tpr = stats[attack]['TPR'][i]
                    plt.loglog(fpr, tpr)
                    legend.append(attack)
                plt.legend(legend)
                plt.xlim(1e-4, 1)
                plt.ylim(1e-4, 1)
                plt.grid(True)
                plt.xlabel('FPR')
                plt.ylabel('TPR')
                plt.title('roc_curve')
                plt.savefig(f'{path}/roc_{running_index}.png')
                running_index += 1