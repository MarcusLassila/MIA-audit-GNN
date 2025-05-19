Implementing black-box node-level membership inference attacks against graph neural networks.

### Instructions

Specify attack settings in config.yaml and run "src/run_audit.py" to audit using membership inference attacks under the specified setting. CSV files of attack performance statistics, ROC data, and plots are stored in results folder.

lood.py computes memorization and information leakage according to the paper [Leave-one-out Distinguishability in Machine Learning](https://arxiv.org/abs/2309.17310).

### Test

run "python -m unittest"
