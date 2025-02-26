import pandas as pd
import matplotlib.pyplot as plt

def parse_roc_csv(name):
    roc_df = pd.read_csv(f'example_results/roc_{name}.csv', sep=',')
    fprs = roc_df.filter(like='FPR_', axis=1)
    tprs = roc_df.filter(like='TPR_', axis=1)

    plt.figure(figsize=(12, 12))
    legend = []
    for (k1, v1), (k2, v2) in zip(fprs.items(), tprs.items()):
        assert k1[4:] == k2[4:]
        plt.loglog(v1.dropna(), v2.dropna())
        legend.append(k1[4:])
    plt.legend(legend)
    plt.xlim(1e-4, 1)
    plt.ylim(1e-4, 1)
    plt.grid(True)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('roc_curve')
    plt.savefig(f'./example_results/roc_{name}.png')

def parse_stat_csv(name):
    stat_df = pd.read_csv(f'example_results/stats_{name}.csv', sep=',')
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(stat_df)

if __name__ == '__main__':
    parse_roc_csv('bayes-optimal-cora-GCN-MI')
    parse_stat_csv('bayes-optimal-cora-GCN-MI')
    parse_roc_csv('bayes-optimal-cora-GCN-MCMC')
    parse_stat_csv('bayes-optimal-cora-GCN-MCMC')
