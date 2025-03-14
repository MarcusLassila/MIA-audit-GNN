import pandas as pd
import matplotlib.pyplot as plt
import argparse

def parse_roc_csv(resdir):
    roc_df = pd.read_csv(f'{resdir}/roc.csv', sep=',')
    fprs = roc_df.filter(like='FPR_0', axis=1)
    tprs = roc_df.filter(like='TPR_0', axis=1)

    plt.figure(figsize=(12, 12))
    legend = []
    for (k1, v1), (k2, v2) in zip(fprs.items(), tprs.items()):
        assert k1[4:] == k2[4:]
        plt.loglog(v1.dropna(), v2.dropna())
        legend.append(k1[6:])
    plt.legend(legend)
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('roc_curve')
    plt.savefig(f'{resdir}/roc.png')

def parse_stat_csv(resdir):
    stat_df = pd.read_csv(f'{resdir}/results.csv', sep=',')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(stat_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--resdir", default="./temp_results", type=str)
    args = parser.parse_args()
    parse_roc_csv(args.resdir)
    parse_stat_csv(args.resdir)
