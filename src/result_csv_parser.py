import utils

import pandas as pd

if __name__ == '__main__':
    df = pd.read_csv('./results/statistics.csv', sep=',').set_index('Unnamed: 0')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(df)
    utils.plot_roc_csv('./results/rocs.csv', './plots')
