import attacks
import datasetup
import hypertuner
import lood
import evaluation
import trainer
import utils

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, index_to_mask
from torchmetrics import Accuracy
from tqdm.auto import tqdm
from collections import defaultdict
from itertools import combinations

class MembershipInferenceExperiment:

    def __init__(self, config):
        self.config = utils.Config(config)
        self.dataset = datasetup.parse_dataset(root=self.config.datadir, name=self.config.dataset)
        self.criterion = Accuracy(task="multiclass", num_classes=self.dataset.num_classes).to(self.config.device)
        print(utils.graph_info(self.dataset))

    def visualize_embedding_distribution(self, target_model, target_samples, attacker, target_fpr, num_hops=2):
        # TODO: plot embeddings
        config = self.config
        true_members = target_samples.train_mask.long().cpu().numpy()
        query_nodes = torch.arange(target_samples.num_nodes)
        emb = evaluation.k_hop_query(
            model=target_model,
            dataset=target_samples,
            query_nodes=query_nodes,
            num_hops=0,
            inductive_split=True,
        )
        soft_preds_0 = attacker.run_attack(target_samples=target_samples, num_hops=0)
        soft_preds_k = attacker.run_attack(target_samples=target_samples, num_hops=num_hops, inductive_inference=True)
        metrics_0 = evaluation.evaluate_binary_classification(preds=soft_preds_0, truth=true_members, target_fpr=target_fpr)
        metrics_k = evaluation.evaluate_binary_classification(preds=soft_preds_k, truth=true_members, target_fpr=target_fpr)
        hard_preds_0 = metrics_0['hard_preds']
        hard_preds_k = metrics_k['hard_preds']
        nodes_0 = torch.tensor((hard_preds_0 & (1 ^ hard_preds_k) & true_members).nonzero()[0], dtype=torch.long)
        nodes_k = torch.tensor((hard_preds_k & (1 ^ hard_preds_0) & true_members).nonzero()[0], dtype=torch.long)
        nodes_of_interest = torch.tensor(((hard_preds_0 | hard_preds_k) & true_members).nonzero()[0], dtype=torch.long)
        labels, counts = torch.unique(target_samples.y[nodes_of_interest], return_counts=True)
        most_vuln_label = labels[counts.argmax()]
        nodes_0_singe_label = nodes_0[target_samples.y[nodes_0] == most_vuln_label]
        nodes_k_singe_label = nodes_k[target_samples.y[nodes_k] == most_vuln_label]

        utils.plot_embedding_2D_scatter(emb)

        def cross_cosine_sim(index_a, index_b, emb):
            avg_cross_sim = 0
            l = 0
            for a in index_a:
                for b in index_b:
                    if a == b:
                        continue
                    avg_cross_sim += F.cosine_similarity(emb[a], emb[b], dim=-1).item()
                    l += 1
            if l == 0:
                return 1.0
            avg_cross_sim /= l
            return avg_cross_sim

        avg_cross_sim = cross_cosine_sim(nodes_0_singe_label, nodes_k_singe_label, emb)
        avg_0_sim = cross_cosine_sim(nodes_0_singe_label, nodes_0_singe_label, emb)
        avg_k_sim = cross_cosine_sim(nodes_k_singe_label, nodes_k_singe_label, emb)
        print(f'Comparing cosine simularities of embeddings for class {most_vuln_label}')
        print(f'average cross sim: {avg_cross_sim:.4f}')
        print(f'average 0 sim: {avg_0_sim:.4f}')
        print(f'average {num_hops} sim: {avg_k_sim:.4f}')

    def visualize_aggregation_effect_on_attack_vulnerabilities(self, attacker, target_samples, target_fpr, num_hops=2, max_num_plotted_nodes=15):
        '''
        Plot test statistic of nodes against decision threshold for nodes that are identified by the 0-hop attack, but not the k-hop attack, or vice versa.
        '''
        config = self.config
        true_members = target_samples.train_mask.long().cpu().numpy()
        soft_preds_0 = attacker.run_attack(target_samples=target_samples, num_hops=0)
        soft_preds_k = attacker.run_attack(target_samples=target_samples, num_hops=num_hops, inductive_inference=True)
        metrics_0 = evaluation.evaluate_binary_classification(preds=soft_preds_0, truth=true_members, target_fpr=target_fpr)
        metrics_k = evaluation.evaluate_binary_classification(preds=soft_preds_k, truth=true_members, target_fpr=target_fpr)
        threshold_0 = metrics_0['threshold']
        threshold_k = metrics_k['threshold']
        hard_preds_0 = torch.from_numpy(metrics_0['hard_preds'])
        hard_preds_k = torch.from_numpy(metrics_k['hard_preds'])
        nodes_of_interest = (hard_preds_0 ^ hard_preds_k).nonzero()
        if nodes_of_interest.nelement() == 0:
            print(f'0-hop and {num_hops}-hop predicted identically.')
            return
        #avg_deg = torch.zeros(size=(nodes_of_interest.shape[0], 2))
        means = torch.zeros(size=(nodes_of_interest.shape[0], 2))
        #classfrac = torch.zeros(size=(nodes_of_interest.shape[0], 2))
        for i, node in enumerate(nodes_of_interest):
            node_index, edge_index, center_node, _ = k_hop_subgraph(
                node_idx=node.item(),
                num_hops=num_hops,
                edge_index=target_samples.edge_index[:, target_samples.inductive_mask],
                relabel_nodes=True,
                num_nodes=target_samples.num_nodes,
            )
            subgraph = Data(
                x=target_samples.x[node_index],
                edge_index=edge_index,
                y=target_samples.y[node_index],
                num_classes=target_samples.num_classes,
            )
            assert torch.allclose(subgraph.x[center_node], target_samples.x[node])
            assert torch.equal(subgraph.y[center_node], target_samples.y[node])
            if hard_preds_0[node]:
                title = '0-hop'
            elif hard_preds_k[node]:
                title = '2-hop'
            else:
                raise
            if true_members[node]:
                title += ' TP'
            else:
                title += ' FP'

            #avg_deg[i, 0] = utils.average_degree(subgraph)
            #avg_deg[i, 1] = utils.average_degree(subgraph)
            #classfrac[i, 0] = (subgraph.y == subgraph.y[center_node]).sum() / subgraph.num_nodes
            #classfrac[i, 1] = (subgraph.y == subgraph.y[center_node]).sum() / subgraph.num_nodes
            means[i, 0] = soft_preds_0[node_index].mean()
            means[i, 1] = soft_preds_k[node_index].mean()
            title += f' [{means[i, 0]:.2f}, {means[i, 1]:.2f}]'
            # utils.plot_k_hop_subgraph(
            #     subgraph,
            #     center_node,
            #     soft_preds_0[node_index],
            #     soft_preds_k[node_index],
            #     hard_preds_0[node_index],
            #     hard_preds_k[node_index],
            #     title,
            # )

        perm_mask = torch.randperm(nodes_of_interest.shape[0])
        node_index = nodes_of_interest[perm_mask][:max_num_plotted_nodes]
        membership_status = true_members[node_index]
        labels = means[perm_mask][:max_num_plotted_nodes]
        preds_0 = soft_preds_0[node_index]
        preds_k = soft_preds_k[node_index]
        indices = torch.arange(1, max_num_plotted_nodes + 1)
        savepath = f'{self.config.savedir}/{self.config.name}_{config.attack}_node_vuln_temp.png'
        low, high, diff = min(threshold_0, threshold_k), max(threshold_0, threshold_k), abs(threshold_0 - threshold_k)
        spread = 2

        plt.figure(figsize=(12, 12))
        plt.scatter(indices, preds_0, label='0-hop')
        plt.scatter(indices, preds_k, label=f'{num_hops}-hop', marker='x')
        plt.plot(indices, torch.ones_like(indices) * threshold_0, label=r'$\gamma$ 0-hop')
        plt.plot(indices, torch.ones_like(indices) * threshold_k, label=rf'$\gamma$ {num_hops}-hop')
        for i in range(max_num_plotted_nodes):
            plt.text(indices[i], preds_0[i], f'{labels[i, 0]:.1f}-{"in" if membership_status[i] else "out"}', fontsize=12, ha='left', va='bottom')
            plt.text(indices[i], preds_k[i], f'{labels[i, 1]:.1f}-{"in" if membership_status[i] else "out"}', fontsize=12, ha='right', va='bottom')
        plt.xticks(np.arange(indices.shape[0]))
        plt.xlabel('Node id')
        plt.ylabel('Test statistic')
        if config.attack == "lira":
            plt.ylim(low - spread * diff, high + spread * diff)
        elif config.attack == "rmia":
            plt.ylim(0.0, 1.0)
        plt.legend()
        utils.savefig_or_show(savepath)

    def analyze_correlation_with_information_leakage(self, attacker, target_samples, target_fpr, num_hops=2):
        config = self.config
        true_members = target_samples.train_mask.long().cpu().numpy()
        soft_preds_0 = attacker.run_attack(target_samples=target_samples, num_hops=0)
        soft_preds_k = attacker.run_attack(target_samples=target_samples, num_hops=num_hops, inductive_inference=True)
        metrics_0 = evaluation.evaluate_binary_classification(preds=soft_preds_0, truth=true_members, target_fpr=target_fpr)
        metrics_k = evaluation.evaluate_binary_classification(preds=soft_preds_k, truth=true_members, target_fpr=target_fpr)
        hard_preds_0 = metrics_0['hard_preds']
        hard_preds_k = metrics_k['hard_preds']

        nodes_of_interest = torch.from_numpy(((hard_preds_0 ^ hard_preds_k) & true_members).nonzero()[0])
        mask = torch.randperm(nodes_of_interest.shape[0])[:50]
        nodes_of_interest = nodes_of_interest[mask]
        preds_0 = soft_preds_0[nodes_of_interest]
        preds_k = soft_preds_k[nodes_of_interest]
        xs = torch.arange(nodes_of_interest.shape[0])
        lood_instance = lood.LOOD(config=self.config)
        leak_0, memo_0 = lood_instance.information_leakage(dataset=target_samples, node_index=nodes_of_interest, num_hops=0)
        leak_k, memo_k = lood_instance.information_leakage(dataset=target_samples, node_index=nodes_of_interest, num_hops=num_hops)

        savedir = config.savedir + '/information_leakage'
        prefix = '_'.join([config.dataset, config.model, config.attack])

        plt.figure(figsize=(12, 12))
        plt.scatter(preds_0, leak_0)
        plt.xlabel('pred')
        plt.ylabel('KL Leakage')
        plt.title('0-hop')
        plt.grid(True)
        savepath=f'{savedir}/{prefix}_leak_0_correlation.png'
        utils.savefig_or_show(savepath)
        plt.figure(figsize=(12, 12))

        plt.scatter(preds_k, leak_k)
        plt.xlabel('pred')
        plt.ylabel('KL Leakage')
        plt.title(f'{num_hops}-hop')
        plt.grid(True)
        savepath=f'{savedir}/{prefix}_leak_{num_hops}_correlation.png'
        utils.savefig_or_show(savepath)

        plt.figure(figsize=(12, 12))
        plt.scatter(preds_0, memo_0)
        plt.xlabel('pred')
        plt.ylabel('Memorization')
        plt.title('0-hop')
        plt.grid(True)
        savepath=f'{savedir}/{prefix}_memo_0_correlation.png'
        utils.savefig_or_show(savepath)

        plt.figure(figsize=(12, 12))
        plt.scatter(preds_k, memo_k)
        plt.xlabel('pred')
        plt.ylabel('Memorization')
        plt.title(f'{num_hops}-hop')
        plt.grid(True)
        savepath=f'{savedir}/{prefix}_memo_{num_hops}_correlation.png'
        utils.savefig_or_show(savepath)

        preds_0, preds_k = utils.min_max_normalization(preds_0, preds_k)
        memo_0, memo_k = utils.min_max_normalization(memo_0, memo_k)
        leak_0, leak_k = utils.min_max_normalization(leak_0, leak_k)

        new_size = 15
        xs = xs[:new_size]
        memo_0 = memo_0[:new_size]
        memo_k = memo_k[:new_size]
        leak_0 = leak_0[:new_size]
        leak_k = leak_k[:new_size]
        preds_0 = preds_0[:new_size]
        preds_k = preds_k[:new_size]

        plt.figure(figsize=(12, 12))
        plt.scatter(xs, memo_0, marker='x', label='memo 0')
        plt.scatter(xs, memo_k, marker='x', label=f'memo {num_hops}')
        plt.scatter(xs, preds_0, marker='o', label='pred 0')
        plt.scatter(xs, preds_k, marker='o', label=f'pred {num_hops}')
        plt.legend()
        plt.xticks(np.arange(new_size))
        plt.grid(True)
        savepath=f'{savedir}/{prefix}_memo_correlation.png'
        utils.savefig_or_show(savepath)

        plt.figure(figsize=(12, 12))
        plt.scatter(xs, leak_0, marker='x', label='leak 0')
        plt.scatter(xs, leak_k, marker='x', label=f'leak {num_hops}')
        plt.scatter(xs, preds_0, marker='o', label='pred 0')
        plt.scatter(xs, preds_k, marker='o', label=f'pred {num_hops}')
        plt.legend()
        plt.xticks(np.arange(new_size))
        plt.grid(True)
        savepath=f'{savedir}/{prefix}_leak_correlation.png'
        utils.savefig_or_show(savepath)

    def train_target_model(self, dataset, plot_training_results=True, compare_with_mlp=True):
        config = self.config

        if self.config.grid_search:
            opt_hyperparams = hypertuner.grid_search(
                dataset=dataset,
                model_type=self.config.model,
                epochs=self.config.epochs_target,
                early_stopping=self.config.early_stopping,
                optimizer=self.config.optimizer,
                inductive_split=config.inductive_split,
            )
            print(f'Grid search results: {opt_hyperparams}')
            lr, weight_decay, dropout, hidden_dim = opt_hyperparams.values()
        else:
            lr, weight_decay, dropout, hidden_dim = config.lr, config.weight_decay, config.dropout, config.hidden_dim_target

        target_model = utils.fresh_model(
            model_type=self.config.model,
            num_features=dataset.num_features,
            hidden_dims=hidden_dim,
            num_classes=dataset.num_classes,
            dropout=dropout,
        )
        train_config = trainer.TrainConfig(
            criterion=self.criterion,
            device=config.device,
            epochs=config.epochs_target,
            early_stopping=config.early_stopping,
            loss_fn=nn.CrossEntropyLoss(reduction='mean'),
            lr=lr,
            weight_decay=weight_decay,
            optimizer=getattr(torch.optim, config.optimizer),
        )
        train_res = trainer.train_gnn(
            model=target_model,
            dataset=dataset,
            config=train_config,
            inductive_split=config.inductive_split
        )
        evaluation.evaluate_graph_training(
            model=target_model,
            dataset=dataset,
            criterion=train_config.criterion,
            inductive_inference=config.inductive_split,
            training_results=train_res if plot_training_results else None,
            plot_title="Target model",
            savedir=config.savedir,
        )
        if compare_with_mlp:
            # Sanity check that graph split is good enough to still give advantage to GNN over MLP.
            mlp_reference_model = utils.fresh_model(
                model_type='MLP',
                num_features=dataset.num_features,
                hidden_dims=hidden_dim,
                num_classes=dataset.num_classes,
                dropout=dropout,
            )
            mlp_train_res = trainer.train_gnn(
                model=mlp_reference_model,
                dataset=dataset,
                config=train_config,
                inductive_split=config.inductive_split
            )
            evaluation.evaluate_graph_training(
                model=mlp_reference_model,
                dataset=dataset,
                criterion=train_config.criterion,
                inductive_inference=config.inductive_split,
                training_results=mlp_train_res if plot_training_results else None,
                plot_title="MLP model",
                savedir=config.savedir,
            )
        return target_model

    def parse_train_stats(self, train_stats):
        return pd.DataFrame({
            'train_acc': [utils.stat_repr(train_stats['train_scores'])],
            'test_acc': [utils.stat_repr(train_stats['test_scores'])],
        }, index=[self.config.name])

    def parse_attack_stats(self, attack_stats):
        config = self.config
        stats = defaultdict(list)
        for key, value in attack_stats.items():
            if isinstance(value[0], float):
                stats[f'{key}'].append(utils.stat_repr(value))
        return pd.DataFrame(stats, index=[config.name])

    def parse_detection_stats(self, detection_stats, tags):
        config = self.config
        detection_table = defaultdict(list)
        detection_counts = np.stack(detection_stats)
        detection_counts_mean = np.mean(detection_counts, axis=0)
        detection_counts_std = np.std(detection_counts, axis=0)
        i = 0
        for r in range(1, len(tags) + 1):
            for idx in combinations(range(len(tags)), r):
                key = '+'.join(tags[j] for j in idx)
                detection_table[key].append(f'{detection_counts_mean[i]:.2f} ({detection_counts_std[i]:.2f})')
                i += 1
        return pd.DataFrame(detection_table, index=[config.name])

    def make_target(self, evaluate_target=True):
        config = self.config
        target_dataset, other_half, target_index, other_index = datasetup.disjoint_graph_split(
            self.dataset,
            train_frac=config.train_frac,
            val_frac=config.val_frac,
            v2=True
        )
        target_model = self.train_target_model(target_dataset)
        if evaluate_target:
            target_scores = {
                'train_score': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=target_dataset,
                    mask=target_dataset.train_mask,
                    criterion=self.criterion,
                    inductive_inference=config.inductive_split,
                ),
                'test_score': evaluation.evaluate_graph_model(
                    model=target_model,
                    dataset=target_dataset,
                    mask=target_dataset.test_mask,
                    criterion=self.criterion,
                    inductive_inference=config.inductive_split,
                ),
            }
        else:
            target_scores = None
        return target_model, target_dataset, other_half, target_index, other_index, target_scores

    def get_attacker(self, target_model, population):
        '''
        Return an instance of the attack class specified by config.attack
        '''
        config = self.config
        match config.attack:
            case "basic-mlp":
                if config.split == 'sampled':
                    shadow_dataset = datasetup.sample_subgraph(
                        self.dataset,
                        num_nodes=self.dataset.num_nodes//2,
                        train_frac=config.train_frac,
                        val_frac=config.val_frac
                    )
                else:
                    shadow_dataset = population.clone()
                attacker = attacks.BasicMLPAttack(
                    target_model=target_model,
                    shadow_dataset=shadow_dataset,
                    config=config,
                )
            case "improved-mlp":
                attacker = attacks.ImprovedMLPAttack(
                    target_model=target_model,
                    population=population,
                    queries=[0, 2],
                    config=config,
                )
            case "confidence":
                attacker = attacks.ConfidenceAttack(
                    target_model=target_model,
                    config=config,
                )
            case "lira":
                # In offline LiRA, the shadow models are trained on datasets that does not contain the target sample.
                # Therefore we make a disjoint split and train shadow models on one part, and attack samples of the other part.
                attacker = attacks.LiRA(
                    target_model=target_model,
                    population=population,
                    config=config,
                )
            case "rmia":
                attacker = attacks.RMIA(
                    target_model=target_model,
                    population=population,
                    config=config,
                )
            case _:
                raise AttributeError(f"No attack named {config.attack}")
        return attacker

    def run_query_experiment(self):
        config = self.config
        train_stats = defaultdict(list)
        attack_stats = defaultdict(list)
        detection_stats = []

        def make_tag(num_hops: int, inductive_flag: bool):
            return f'{num_hops}{"I" if inductive_flag else "T"}'
        
        def query_generator():
            for num_hops in config.query_hops:
                if num_hops == 0:
                    yield (num_hops, True)
                elif config.inductive_inference is None:
                    for p in True, False:
                        yield (num_hops, p)
                else:
                    yield (num_hops, config.inductive_inference)

        for i_experiment in range(1, config.num_experiments + 1):
            print(f'Running experiment {i_experiment}/{config.num_experiments}.')

            # Use fixed random seeds such that each experimental configuration is evaluated on the same dataset
            set_seed(i_experiment)

            tags = [] # Used to label attack setups.
            # Train and evaluate target model.
            target_model, target_dataset, other_half, _, _, target_scores = self.make_target()
            train_stats['train_scores'].append(target_scores['train_score'])
            train_stats['test_scores'].append(target_scores['test_score'])

            attacker = self.get_attacker(target_model=target_model, population=other_half)

            # Remove validation mask from target samples
            if target_dataset.val_mask.sum().item() > 0:
                target_samples = datasetup.masked_subgraph(target_dataset, ~target_dataset.val_mask)
            else:
                target_samples = target_dataset.clone()
            truth = target_samples.train_mask.long()

            if config.num_experiments == 1:
                #self.visualize_embedding_distribution(target_model, target_samples, attacker, config.target_fpr, num_hops=2)
                #self.analyze_correlation_with_information_leakage(attacker, target_samples, config.target_fpr, num_hops=2)
                self.visualize_aggregation_effect_on_attack_vulnerabilities(attacker, target_samples, config.target_fpr)
                pass

            soft_preds = []
            true_positives = []

            # Run attack using the specified set of k-hop neighborhood queries.
            for num_hops, inductive_flag in query_generator():
                tag = make_tag(num_hops, inductive_flag)
                tags.append(tag)
                preds = attacker.run_attack(target_samples=target_samples, num_hops=num_hops, inductive_inference=inductive_flag)
                metrics = evaluation.evaluate_binary_classification(preds, truth, config.target_fpr)
                soft_preds.append(preds)
                fpr, tpr = metrics['ROC']
                true_positives.append(metrics['TP'])
                attack_stats[f'{tag}_FPR'].append(fpr)
                attack_stats[f'{tag}_TPR'].append(tpr)
                attack_stats[f'{tag}_AUC'].append(metrics['AUC'])
                attack_stats[f'{tag}_TPR@{config.target_fpr}FPR'].append(metrics['TPR@FPR'])

            if len(soft_preds) > 1:
                soft_preds = torch.stack(soft_preds)
                multi_tpr, _ = utils.tpr_at_fixed_fpr_multi(soft_preds, truth, config.target_fpr)
                attack_stats[f'TPR@{config.target_fpr}FPR_multi'].append(multi_tpr)

            detection_counts = np.array(evaluation.inclusions(true_positives), dtype=np.int64)
            detection_stats.append(detection_counts)

        if config.make_roc_plots:
            for tag in tags:
                savepath = f'{config.savedir}/{config.name}_{tag}_roc_loglog.png'
                fprs = attack_stats[f'{tag}_FPR']
                tprs = attack_stats[f'{tag}_TPR']
                utils.plot_multi_roc_loglog(fprs, tprs, savepath=savepath)

        train_stats_df = self.parse_train_stats(train_stats)
        attack_stats_df = self.parse_attack_stats(attack_stats)
        detection_df = self.parse_detection_stats(detection_stats, tags)
        return train_stats_df, attack_stats_df, detection_df

    def run_monte_carlo_inference_experiment(self):
        config = self.config
        train_stats = defaultdict(list)
        attack_stats = defaultdict(list)

        for i_experiment in range(1, config.num_experiments + 1):
            print(f'Running experiment {i_experiment}/{config.num_experiments}.')

            # Use fixed random seeds such that each experimental configuration is evaluated on the same dataset
            set_seed(i_experiment)

            # Train and evaluate target model.
            target_model, target_dataset, other_half, _, _, target_scores = self.make_target()
            train_stats['train_scores'].append(target_scores['train_score'])
            train_stats['test_scores'].append(target_scores['test_score'])

            attacker = self.get_attacker(target_model=target_model, population=other_half)

            # Remove validation mask from target samples
            if target_dataset.val_mask.sum().item() > 0:
                target_samples = datasetup.masked_subgraph(target_dataset, ~target_dataset.val_mask)
            else:
                target_samples = target_dataset.clone()
            truth = target_samples.train_mask.long()

            # Approximate training mask
            preds = attacker.run_attack(target_samples=target_samples, num_hops=0)
            metrics = evaluation.evaluate_binary_classification(preds, truth, config.target_fpr)
            fpr, tpr = metrics['ROC']
            attack_stats[f'0hop_FPR'].append(fpr)
            attack_stats[f'0hop_TPR'].append(tpr)
            attack_stats[f'0hop_AUC'].append(metrics['AUC'])
            attack_stats[f'0hop_TPR@{config.target_fpr}FPR'].append(metrics['TPR@FPR'])

            n = 200
            sorted_preds = preds.argsort(dim=0, descending=True)
            # for n in tqdm(range(50, 601, 50), desc='testing n'):
            print('n =', n)
            top_n = sorted_preds[:n]
            bottom_n = sorted_preds[-n:]
            top_mask = index_to_mask(top_n, size=target_samples.num_nodes)
            bottom_mask = index_to_mask(bottom_n, size=target_samples.num_nodes)
            print(f'correct in: {(top_mask & target_samples.train_mask).sum().item() / n:.4f}')
            print(f'correct out: {(bottom_mask & target_samples.test_mask).sum().item() / n:.4f}')

            approximate_edge_mask = []
            for a, b in target_samples.edge_index.T:
                approximate_edge_mask.append(top_mask[a] and top_mask[b] or bottom_mask[a] and bottom_mask[b])
            approximate_edge_mask = torch.tensor(approximate_edge_mask)

            simplified_target_samples = target_samples.clone()
            simplified_target_samples.edge_index = simplified_target_samples.edge_index[:, approximate_edge_mask]
            print(target_samples.edge_index.shape[1], simplified_target_samples.edge_index.shape[1])
            assert target_samples.edge_index.shape[1] > simplified_target_samples.edge_index.shape[1]

            monte_carlo_masks = []
            #monte_carlo_masks.append(torch.zeros(simplified_target_samples.edge_index.shape[1], dtype=torch.bool)) # Always include the 0-hop query
            for _ in range(config.mc_inference_samples):
                #monte_carlo_masks.append(approximate_edge_mask)
                monte_carlo_masks.append(datasetup.random_edge_mask(simplified_target_samples, 0.8))

            preds = attacker.run_attack(target_samples=simplified_target_samples, inductive_inference=False, monte_carlo_masks=monte_carlo_masks)
            metrics = evaluation.evaluate_binary_classification(preds, truth, config.target_fpr)
            fpr, tpr = metrics['ROC']
            print(f'{n}: {metrics['TPR@FPR']:.4f}\n')
            attack_stats[f'FPR'].append(fpr)
            attack_stats[f'TPR'].append(tpr)
            attack_stats[f'AUC'].append(metrics['AUC'])
            attack_stats[f'TPR@{config.target_fpr}FPR'].append(metrics['TPR@FPR'])

            preds = attacker.run_attack(target_samples=target_samples, num_hops=2, inductive_inference=True)
            metrics = evaluation.evaluate_binary_classification(preds, truth, config.target_fpr)
            fpr, tpr = metrics['ROC']
            attack_stats[f'2hop_FPR'].append(fpr)
            attack_stats[f'2hop_TPR'].append(tpr)
            attack_stats[f'2hop_AUC'].append(metrics['AUC'])
            attack_stats[f'2hop_TPR@{config.target_fpr}FPR'].append(metrics['TPR@FPR'])

        if config.make_roc_plots:
            savepath = f'{config.savedir}/{config.name}_roc_loglog.png'
            fprs = attack_stats[f'FPR']
            tprs = attack_stats[f'TPR']
            utils.plot_multi_roc_loglog(fprs, tprs, savepath=savepath)

        train_stats_df = self.parse_train_stats(train_stats)
        attack_stats_df = self.parse_attack_stats(attack_stats)
        return train_stats_df, attack_stats_df

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def main(config):
    set_seed(config['seed'])
    config['dataset'] = config['dataset'].lower()
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    mie = MembershipInferenceExperiment(config)
    match config['experiment']:
        case 'query':
            return mie.run_query_experiment()
        case 'mc-inference':
            return mie.run_monte_carlo_inference_experiment()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack", default="confidence", type=str)
    parser.add_argument("--dataset", default="cora", type=str)
    parser.add_argument("--split", default="sampled", type=str)
    parser.add_argument("--inductive-split", action=argparse.BooleanOptionalAction)
    parser.add_argument("--inductive-inference", action=argparse.BooleanOptionalAction)
    parser.add_argument("--model", default="GCN", type=str)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--epochs-target", default=20, type=int)
    parser.add_argument("--epochs-shadow", default=20, type=int)
    parser.add_argument("--epochs-mlp-attack", default=100, type=int)
    parser.add_argument("--grid-search", action=argparse.BooleanOptionalAction)
    parser.add_argument("--lr", default=1e-2, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--dropout", default=0.5, type=float)
    parser.add_argument("--early-stopping", default=0, type=int)
    parser.add_argument("--hidden-dim-target", default=[64], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--hidden-dim-mlp-attack", default=[256, 64], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--query-hops", default=[0], type=lambda x: [*map(int, x.split(','))])
    parser.add_argument("--experiment", default="query", type=str)
    parser.add_argument("--num-experiments", default=1, type=int)
    parser.add_argument("--target-fpr", default=0.01, type=float)
    parser.add_argument("--optimizer", default="Adam", type=str)
    parser.add_argument("--num-shadow-models", default=64, type=int)
    parser.add_argument("--rmia-gamma", default=2.0, type=float)
    parser.add_argument("--mc-inference-samples", default=0, type=int)
    parser.add_argument("--name", default="unnamed", type=str)
    parser.add_argument("--datadir", default="./data", type=str)
    parser.add_argument("--savedir", default="./results", type=str)
    parser.add_argument("--train-frac", default=0.5, type=float)
    parser.add_argument("--val-frac", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    config = vars(args)
    config['make_roc_plots'] = True
    config['name'] = config['dataset'] + '_' + config['model']
    if config['inductive_split'] is None:
        config['inductive_split'] = True
    if config['inductive_inference'] is None:
        config['inductive_inference'] = True
    print('Running MIA experiment.')
    print(utils.Config(config))
    print()

    ret = main(config)
    if config['experiment'] == 'query':
        train_df, stat_df, detection_df = ret
    elif config['experiment'] == 'mc-inference':
        train_df, stat_df = ret

    pd.set_option('display.max_columns', 500)
    print('Target training statistics:')
    print(train_df)
    print('Attack performance statistics:')
    print(stat_df)
    if len(config['query_hops']) > 1:
        print('Node detection count set differences:')
        print(detection_df)
