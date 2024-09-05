Replicate the black-box node-level membership inference attacks of [Membership Inference Attack on Graph Neural Networks](https://arxiv.org/abs/2101.06570), [Quantifying Privacy Leakage in Graph Embedding](https://arxiv.org/abs/2010.00906) and [Node-Level Membership Inference Attacks Against Graph Neural Networks
](https://arxiv.org/abs/2102.05429), with the purpose of re-evaluating the results, particurlarly at low FPR. Also implements the "LiRA" membership inference attack from [Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570). 

### Instructions

Specify attack settings in config.yaml and run "src/run_experiments.py". CSV files of attack performance statistics, ROC data, and plots are stored in results folder. 

Alternatively, run "src/run_mia.py" which only run one MIA simulation experiment and can be customized with the following flags:

* --attack: Type of attack to use.
    * "basic-mlp": The black-box shadow model attack used in "Membership Inference Attack on Graph Neural Networks", "Quantifying Privacy Leakage in Graph Embedding" and "Node-Level Membership Inference Attacks Against Graph Neural Networks".
    * "confidence": The confidence attack in "Quantifying Privacy Leakage in Graph Embedding", which thresholds the confidence values from the target model for a membership prediction.
    * "lira": The offline version of LiRA from "Membership Inference Attacks From First Principles".
    * "rmia": The attack from "Low-Cost High-Power Membership Inference Attacks" (only offline version currently).
* --dataset: cora, corafull, citeseer, chameleon, pubmed or flickr.
* --split: How to split the dataset into target/shadow datasets.
    * "sampled": randomly sample subgraphs consisting of 50% of the nodes for target and shadow model. Overlap allowed.
    * "disjoint": randomly split the graph into two disjoint halves, one for the target and the other for the shadow model.
* --transductive: Enable transductive learning, i.e. the entire graph (training, validation and test nodes) is used during training. The default is to perform inductive learning, where validation and test sets are split in disconnected subgraphs.
* --inductive-inference: Make inferences using the inductive split. This ensures that member nodes are only connected to non-member nodes and vice versa.
* --model: GCN, DecoupledGCN, GCNConv (single layer), SGC, GraphSAGE, GAT or GIN (case sensitive).
* --epochs_target: Number of epochs to train target and shadow model.
* --epochs-attack: Number of epochs to train the attack model.
* --batch-size: Size of mini-batches.
* --lr: Learning rate when training target and shadow models.
* --weight_decay: Weight decay (L2 penalty) to when training traget and shadow models.
* --dropout: Dropout rate when training target and shadow model.
* --early-stopping: Enable early stopping when training target, shadow and attack model.
* --hidden-dim-target: Dimension of the hidden layer of the 2-layer GNN target/shadow model.
* --hidden-dim-attack: Dimensions of the hidden layers in the MLP. Input is given as a comma separeted list, e.g. 128,64,32.
* --rmia-offline-interp-param: Hyperparamter for interpolation of "in models" from "out models".
* --query-hops: List of the k-hops to use during inference in the attacks.
* --experiments: Number of times to repeat the whole attack experiment (including retraining target model) and average results over.
* --optimizer: Will call getattr(torch.optim, optimizer) so it better exist in torch.optim.
* --num-shadow-models: For LiRA.
* --datadir: Path to save dataset.
* --savedir: Path to store results.
* --name: Name to add to result files.
* --experiments: Number of samples (retraining target, shadow and attack models) to compute result statistics over.
