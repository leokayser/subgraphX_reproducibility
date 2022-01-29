import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Callable, List, Dict, Union, Tuple, Any
from enum import Enum
import pprint

import networkx as nx
from matplotlib import pyplot as plt

import numpy as np
import torch
import torch_geometric.datasets
from torch import optim
from torch.nn import ReLU, Linear, Softmax
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, Sequential, GNNExplainer

from src.algorithm.subgraph_x import SubgraphX
from src.utils.logging import load_data, save_data, aggregate_fidelity_sparsity, compute_avg_runtime, save_str
from src.utils.metrics import sparsity, fidelity
from src.utils.task_enum import Task, Experiment
from src.utils.training import train_emb, train_model, test
from src.utils.utils import get_device, set_seed, convert_edge_mask_to_subset
from src.utils.visualization import plot_results, plot_search_tree
import src.transposition.karate_club as karate_club
from src.parameter_experiments.parameters import SubgraphXSnapshot, get_sx_params, get_experiment_params
from src.algorithm.shapley import fidelity_wrapper


def prepare_experiment():
    dataset = torch_geometric.datasets.KarateClub()
    graph = dataset.data
    device = get_device()

    emb_model = karate_club.train_or_load_embedding(graph, only_load=True)
    graph, train_loader, val_loader, test_loader = karate_club.prepare_dataset(emb_model)

    model, loss_func = karate_club.train_or_load_gcn(train_loader, val_loader, only_load=True)
    test_loss, test_acc = test(model, False, test_loader, loss_func, task=Task.NODE_CLASSIFICATION)
    print(f'test loss: {test_loss}, test_acc: {test_acc}')

    return model, test_loader


def run_experiment(name, experiment_params, sx_params, graph_loader, seed=0):
    set_seed(seed)  # IMPORTANT!!

    path = os.path.join(experiment_params['base_dir'], name)
    snapshot_after = experiment_params['snapshot_after']
    n_mins = experiment_params['n_mins']
    sx_params['m'] = snapshot_after[-1]

    print('\nStarting experiment with parameters')
    print('\n'.join([f'\t{k} = {v}' for k, v in sx_params.items() if k != 'model']))
    print(f'\tsnapshot_after = {snapshot_after}')
    # print(f'\tn_mins = {n_mins}')
    print(f'\tbasename =  {os.path.basename(path)}')
    print()

    # nodes of test set
    test_graph = next(iter(graph_loader))
    test_mask = test_graph.test_mask
    test_idx = torch.arange(0, len(test_mask))[test_mask].tolist()
    # test_idx = [1]  #, 3]  # WE WANT SOME RESULTS, FAST

    # snapshots = load_data(path)
    snapshots = dict()
    for node in test_idx:
        snapshots[node] = dict()

    # test saving results
    if os.path.isfile(path):
        print('Warning: overwriting old snapshots!')
        time.sleep(3)
    print('Saving works ...')
    save_data(path, snapshots)

    for node in test_idx:
        print(f'Looking at node {node} ({test_idx.index(node)+1}/{len(test_idx)})')
        nodes_to_keep = [node]

        # create subgraphx thing
        subgraphx = SubgraphX(**sx_params)
        #subgraphx = SubgraphX(model, num_layers=2, exp_weight=5, m=30, t=50, task=Task.NODE_CLASSIFICATION)
        mcts = subgraphx.generate_mcts(test_graph, n_min=min(n_mins), nodes_to_keep=nodes_to_keep,
                                       exhaustive=True)

        total_time = 0

        for i in range(1, snapshot_after[-1]+1):
            # do mcts iteration
            start_time = time.time()
            mcts.search_one_iteration()
            #explanation_set, mcts = None, None #subgraphx(test_graph, n_min=n_min, nodes_to_keep=[node])
            end_time = time.time()
            total_time += end_time - start_time

            # make snapshot if necessary
            if i in snapshot_after:
                explanations = []
                for n in n_mins:
                    explanation_set = mcts.best_node(n).node_set
                    sparsity_score = sparsity(test_graph, explanation_set)
                    fidelity_score = fidelity(test_graph, explanation_set, sx_params["model"], task=sx_params["task"],
                                              nodes_to_keep=nodes_to_keep)
                    explanation_result = (explanation_set, sparsity_score, fidelity_score)

                    search_tree = mcts.search_tree_to_networkx()

                    explanations.append(explanation_result)
                snap = SubgraphXSnapshot(index=node, explanations=explanations, search_tree=search_tree,
                                         iteration_no=i, timestamp=total_time)
                snapshots[node][i] = snap

    print(f'Experiment took {sum([snap.timestamp for _,snaps in snapshots.items() for _,snap in snaps.items()]):.2f} seconds')
    save_data(path, snapshots)


def main():
    base_dir = './result_data/karate_club/experiments'
    model, graph_loader = prepare_experiment()
    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20])

    # experiments with 20 iterations of mcts (same as in the paper)
    sx_params = get_sx_params(model, t=100)
    run_experiment('m20t100', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params(model, t=20)
    run_experiment('m20t20', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params(model, t=5)
    run_experiment('m20t5', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params(model, value_func=fidelity_wrapper)
    run_experiment('m20_score_fidelity', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params(model, experiment=Experiment.GREEDY, t=100, max_children=-1)
    run_experiment('greedy_shapley', exp_params, sx_params, graph_loader)

    sx_params = get_sx_params(model, value_func=fidelity_wrapper, experiment=Experiment.GREEDY, max_children=-1)
    run_experiment('greedy_fidelity', exp_params, sx_params, graph_loader)

    # experiments with custom iteration counts
    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20, 50, 100, 200, 500, 1000])
    sx_params = get_sx_params(model, value_func=fidelity_wrapper, experiment=Experiment.RANDOM, max_children=-1)
    run_experiment('m1000_random', exp_params, sx_params, graph_loader)

    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1])
    sx_params = get_sx_params(model, value_func=fidelity_wrapper, experiment=Experiment.GREEDY, max_children=-1)
    run_experiment('greedy_one', exp_params, sx_params, graph_loader)

    exp_params = get_experiment_params(base_dir=base_dir, snapshot_after=[1, 5, 10, 15, 20, 25, 30])
    sx_params = get_sx_params(model, t=50, max_children=-1)
    run_experiment('m30t50', exp_params, sx_params, graph_loader)


if __name__ == '__main__':
    set_seed(0)  # IMPORTANT!!
    main()
