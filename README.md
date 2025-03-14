Implementing black-box node-level membership inference attacks against graph neural networks.

### Instructions

Specify attack settings in config.yaml and run "src/run_audit.py" to audit using membership inference attacks under the specified setting. CSV files of attack performance statistics, ROC data, and plots are stored in results folder.

lood.py computes memorization and information leakage according to the paper [Leave-one-out Distinguishability in Machine Learning](https://arxiv.org/abs/2309.17310).

mia_audit.py can be run directly with command line arguments specifying the configuration. See source file for parameters and default values.

### Test

run "python -m unittest"
