Replicate the black-box node-level membership inference attacks of [Membership Inference Attack on Graph Neural Networks](https://arxiv.org/abs/2101.06570), [Quantifying Privacy Leakage in Graph Embedding](https://arxiv.org/abs/2010.00906) and [Node-Level Membership Inference Attacks Against Graph Neural Networks
](https://arxiv.org/abs/2102.05429), with the purpose of re-evaluating the results, particurlarly at low FPR. Also implements the "LiRA" membership inference attack from [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570). 

### Instructions

Specify attack settings in config.yaml and run "src/run_experiments.py". CSV files of attack performance statistics, ROC data, and plots are stored in results folder. 

Alternatively, run "src/run_mia.py" which only run one MIA simulation experiment and can be customized with the following flags:

* --attacks: Comma separated string specifying the attacks to run experiment on.
    * "basic-mlp": The black-box shadow model attack used in "Membership Inference Attack on Graph Neural Networks", "Quantifying Privacy Leakage in Graph Embedding" and "Node-Level Membership Inference Attacks Against Graph Neural Networks".
    * "confidence": The confidence attack in "Quantifying Privacy Leakage in Graph Embedding", which thresholds the confidence values from the target model for a membership prediction.
    * "lira": The offline version of LiRA from "Membership Inference Attacks From First Principles".
    * "rmia": The attack from "Low-Cost High-Power Membership Inference Attacks" (only offline version currently).
* --dataset: amazon-computers, amazon-photo, cora, corafull, citeseer, chameleon, pubmed, flickr or reddit.
* --split: How to split the dataset into target/shadow datasets.
    * "sampled": randomly sample subgraphs consisting of 50% of the nodes for target and shadow model. Overlap allowed.
    * "disjoint": randomly split the graph into two disjoint halves, one for the target and the other for the shadow model.
* --no-inductive_split: Use transductive training split, i.e. interconnections between training, validation and test set are kept.
* --no-inductive-inference: Make inferences using the transductive training split.
* --model: GCN, DecoupledGCN, GCNConv (single layer), SGC, GraphSAGE, GAT or GIN (case sensitive).
* --epochs_target: Number of epochs to train target and shadow model.
* --epochs-mlp-attack: Number of epochs to train the attack model in basic-mlp attack.
* --batch-size: Size of mini-batches.
* --lr: Learning rate when training target and shadow models.
* --weight_decay: Weight decay (L2 penalty) to when training traget and shadow models.
* --dropout: Dropout rate when training target and shadow model.
* --early-stopping: Number of successive epochs with no performance improvement on validation set before stopping the training. If <= 0 no early stopping is used. If 0 the model of the last training epoch will be used, otherwise the model with best validation set perfomance is used.
* --hidden-dim-target: Dimension of the hidden layer of the 2-layer GNN target/shadow model.
* --hidden-dim-mlp-attack: Dimensions of the hidden layers in the MLP of the basic-mlp attack model. Input is given as a comma separeted list, e.g. 128,64,32.
* --rmia-gamma: Threshold on likelihood ratio to count as evidence of membership in RMIA attack (see parameter gamma in "Low-Cost High-Power Membership Inference Attacks" paper).
* --query-hops: List of the k-hops to use during inference in the attacks.
* --experiments: Number of times to repeat the whole attack experiment (including retraining target model) and average results over.
* --optimizer: Will call getattr(torch.optim, optimizer) so it better exist in torch.optim.
* --num-shadow-models: For LiRA.
* --datadir: Path to save dataset.
* --savedir: Path to store results.
* --name: Name to add to result files.
* --experiments: Number of samples (retraining target, shadow and attack models) to compute result statistics over.
* --train-frac: Fraction of training nodes, applicable to all graphs used for training (e.g. target graph, shadow graphs etc.).
* --val-frac: Fraction of validation nodes, applicable to all graphs used for training (e.g. target graph, shadow graphs etc.).

### Test

run "python -m unittest"

### TODO

1. Add support for PPI and other multi graph datasets.
2. Extend 0 vs 2-hop test statistic visualization. (Mostly done)
3. Refactor membership inference experiment. (Mostly done)
4. Rethink how inductive training split is handled (should trainer copy dataset?). (Mostly done)
5. Replace printouts by logger?
6. Stop using validation set?
