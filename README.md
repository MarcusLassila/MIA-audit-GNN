Implementing black-box node-level membership inference attacks against graph neural networks.

### Instructions

Specify attack settings in config.yaml and run "src/run_experiments.py". CSV files of attack performance statistics, ROC data, and plots are stored in results folder. 

run_mia.py contains attack experiments using offline attacks, splitting graphs into a disjoint target graph and a population graph.

run_mia_2.py contains attack experiments using online attacks in the simplified graph attack game.

lood.py computes memorization and information leakage according to the paper [Leave-one-out Distinguishability in Machine Learning](https://arxiv.org/abs/2309.17310).

See command line arguments in run_mia.py and run_mia_2.py for options when running these experiments directly.

### Test

run "python -m unittest"
