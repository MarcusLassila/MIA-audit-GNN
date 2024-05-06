Replicate the results of node-level membership inference attacks of [Membership Inference Attack on Graph Neural Networks](https://arxiv.org/abs/2101.06570) and [Node-Level Membership Inference Attacks Against Graph Neural Networks
](https://arxiv.org/abs/2102.05429), with the purpose of re-evaluating the results, particurlarly at low FPR.

### Instructions

Specify attack settings in config.yaml and run "src/run_experiments.py". Plots of ROC curves and training results are stored in plots, and averaged metrics are written to MIA_output.yaml.

Alternatively, run "src/run_mia.py" which only run one MIA simulation experiment and can be customized by the following flags:

* --dataset: cora, siteseer, pubmed or flickr. Default cora.
* --model: GCN, SGC, SAGE or GAT. Default GCN.
* --epochs: Number of epochs to train target and shadow model. Default 50.
* --epochs-attack: Number of epochs to train the attack model. Default 100.
* --batch-size: Size of mini-batches. Default 32.
* --lr: Learning rate when training target and shadow model (attack model training is fixed to 0.001 currently). Default 0.001. 
* --datadir: Path to save dataset. Default './data'.
* --savedir: Path to store plots. Default './plots'.
* --name: Name to add to result files. Default 'unnamed'.
* --experiments: Number of samples (retraining target, shadow and attack models) to compute result statistics over. Default 1.
