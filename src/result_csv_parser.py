import utils

import argparse
import pandas as pd
import re

def merge_mean_and_std(row):
    prefixes = {match.group(1) for idx in row.index for match in re.finditer(r'(.*)_(mean|std)$', idx)}
    for prefix in prefixes:
        row[prefix] = f'{row[prefix + "_mean"]} ({row[prefix + "_std"]})'
        row = row.drop([prefix + "_mean", prefix + "_std"])
    return row

def subdivide_stats_csv(fpr):
    df = pd.read_csv('./results/statistics.csv', sep=',').set_index('Unnamed: 0')
    df_1 = df[['train_acc_mean', 'train_acc_std', 'test_acc_mean', 'test_acc_std']].apply(merge_mean_and_std, axis=1).sort_index(axis=1, key=lambda x: x.str.lower(), ascending=False)
    df_2 = df.loc[:, df.columns.str.startswith('auroc')].apply(merge_mean_and_std, axis=1).sort_index(axis=1, key=lambda x: x.str.lower())
    df_3 = df.loc[:, df.columns.str.startswith('tpr_')].apply(merge_mean_and_std, axis=1).sort_index(axis=1, key=lambda x: x.str.lower())
    print(df_1)
    print(df_2)
    print(df_3)
    df_1.to_csv('./results/statistics_train.csv', sep=',', index=True, index_label='')
    df_2.to_csv('./results/statistics_auroc.csv', sep=',', index=True, index_label='')
    df_3.to_csv('./results/statistics_TPR.csv', sep=',', index=True, index_label='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fpr", default="0.001", type=str)
    args = parser.parse_args()
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    subdivide_stats_csv(args.fpr)
