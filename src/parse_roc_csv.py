import pandas as pd
import matplotlib.pyplot as plt

def parse_roc_csv():
    roc_df = pd.read_csv('results/roc.csv', sep=',')
    fprs = roc_df.filter(like='FPR_', axis=1)
    tprs = roc_df.filter(like='TPR_', axis=1)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(fprs)
    print(tprs)

if __name__ == '__main__':
    parse_roc_csv()