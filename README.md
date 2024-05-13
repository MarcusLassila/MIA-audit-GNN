Replicate the node-level membership inference attacks of [Membership Inference Attack on Graph Neural Networks](https://arxiv.org/abs/2101.06570) and [Node-Level Membership Inference Attacks Against Graph Neural Networks
](https://arxiv.org/abs/2102.05429), with the purpose of re-evaluating the results, particurlarly at low FPR.

### Instructions

Specify attack settings in config.yaml and run "src/run_experiments.py". CSV files of attack performance statistics, ROC data, and plots are stored in results folder. 

Alternatively, run "src/run_mia.py" which only run one MIA simulation experiment and can be customized with the following flags:

* --dataset: cora, corafull, citeseer, pubmed or flickr.
* --split: How to split the dataset into target/shadow datasets.
    * "sampled": randomly sample subgraphs consisting of 50% of the nodes for target and shadow model. Overlap allowed.
    * "disjoint": randomly split the graph into two disjoint halves, one for the target and the other for the shadow model.
    * "TSTF": Randomly apply a train, val, test split of the data, i.e. "Train on Subset, Test on Full". Two different splits for target and shadow models.
* --model: GCN, SGC, GraphSAGE, GAT or GIN (case sensitive).
* --epochs_target: Number of epochs to train target and shadow model.
* --epochs-attack: Number of epochs to train the attack model.
* --batch-size: Size of mini-batches.
* --lr: Learning rate when training target and shadow model (attack model training is fixed to 0.001 currently).
* --dropout: Dropout rate when training target and shadow model.
* --early-stopping: Enable early stopping when training target, shadow and attack model.
* --hidden-dim-target: Dimension of the hidden layer of the 2-layer GNN target/shadow model.
* --hidden-dim-attack: Dimensions of the hidden layers in the MLP. Input is given as a comma separeted list, e.g. 128,64,32.
* --query-hops: The size of the k-hop neighborhood to query the target model when creating features for the attack model.
* --datadir: Path to save dataset.
* --savedir: Path to store results.
* --plot-roc: Plot ROC curve in log scale and store in savedir.
* --name: Name to add to result files.
* --experiments: Number of samples (retraining target, shadow and attack models) to compute result statistics over.
