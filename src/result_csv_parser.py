import utils

import pandas as pd

def subdivide_stats_csv():
    df = pd.read_csv('./results/statistics.csv', sep=',').set_index('Unnamed: 0')
    df_1 = df[['train_acc_mean', 'train_acc_std', 'test_acc_mean', 'test_acc_std']]
    df_2 = df[['auroc_0_mean', 'auroc_0_std', 'auroc_2_II_mean', 'auroc_2_II_std', 'auroc_2_TI_mean', 'auroc_2_TI_std']]
    df_3 = df[['tpr_0.005_fpr_0_mean', 'tpr_0.005_fpr_0_std', 'tpr_0.005_fpr_2_II_mean', 'tpr_0.005_fpr_2_II_std', 'tpr_0.005_fpr_2_TI_mean', 'tpr_0.005_fpr_2_TI_std']]
    df_4 = df[['tpr_0.005_fpr_combined_mean', 'tpr_0.005_fpr_combined_std']]
    print(df_1)
    print(df_2)
    print(df_3)
    print(df_4)
    df_1.to_csv('./results/statistics_train.csv', sep=',', index=True, index_label='')
    df_2.to_csv('./results/statistics_auroc.csv', sep=',', index=True, index_label='')
    df_3.to_csv('./results/statistics_TPR.csv', sep=',', index=True, index_label='')
    df_4.to_csv('./results/statistics_combined.csv', sep=',', index=True, index_label='')

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    subdivide_stats_csv()
