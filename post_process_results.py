import sys
import pickle
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

if __name__ == '__main__':
    assert len(sys.argv) == 3, "Usage: python post_process_results.py root audit-name"
    path = Path(sys.argv[1] + '/' + sys.argv[2])
    name = sys.argv[2]
    stats_list = []
    for pickle_file in path.glob('stats_*.pkl'):
        with open(pickle_file, 'rb') as f:
            stats = pickle.load(f)
            stats_list.append(stats)
    frames = []
    for attack in stats_list[0].keys():
        table = defaultdict(list)
        for key in stats_list[0][attack].keys():
            if key in ('TPR', 'FPR'):
                continue
            combined_value = []
            for stats in stats_list:
                value = stats[attack][key]
                combined_value.append(value)
            combined_value = np.concatenate(combined_value)
            stat = f'{combined_value.mean():.4f} ({combined_value.std():.4f})'
            table[key].append(stat)
        frames.append(pd.DataFrame(table, index=[name + '_' + attack]))
    stat_df = pd.concat(frames)
    stat_df.to_csv(f'{path}/recombined_results.csv', sep=',')
