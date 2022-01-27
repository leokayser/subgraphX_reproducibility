import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Callable, List, Dict, Union, Tuple, Any
from enum import Enum

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
from src.utils.task_enum import Task
from src.utils.training import train_emb, train_model, test
from src.utils.utils import get_device, set_seed, convert_edge_mask_to_subset
from src.utils.visualization import plot_results, plot_search_tree
import src.transposition.karate_club as karate_club


@dataclass
class SubgraphXSnapshot:
    """Stores a snapshot of the MCTS and explanations.

    Attributes:
        index           What this snapshot explains (e.g. node id or graph idx)
        iteration_no    Iteration of MCTS
        explanations    List of tuples (explanation-set, sparsity, fidelity)
        search_tree     Graph that contains MCTS search paths
        timestamp       Total time it took to get to this snapshot (in seconds)
    """
    index: Any
    iteration_no: int
    explanations: List[ Tuple[ Set[int], float, float] ]
    search_tree: nx.DiGraph
    timestamp: float


def plot_experiment(ax, data, nodes, iterations, n_mins, label, n, m, n_min):
    data = np.array(data)  # shape (nodes, iterations, n_mins)

    def is_axis(a):
        return a == 'x' or a == 'y'

    if n is None:
        data = np.mean(data, axis=0, keepdims=True)
    elif is_axis(n):
        x = nodes
    else:
        n_idx = nodes.index(n)
        data = data[[n_idx],:,:]

    if m is None:
        raise Exception("Averaging along iterations is a bad idea.")
    elif is_axis(m):
        x = iterations
    else:
        m_idx = iterations.index(m)
        data = data[:, [m_idx], :]

    if n_min is None:
        data = np.mean(data, axis=2, keepdims=True)
    elif is_axis(n_min):
        x = 'sparsity'
    else:
        n_min_idx = n_mins.index(n_min)
        data = data[:, :, [n_min_idx]]

    data = np.squeeze(data)  # down to [ (fid, spar), (fid, spar), ... ]

    if x == 'sparsity':
        x = data[:, 1]

    y = data[:, 0]  # down to [ fid, fid, fid, ... ]
    if len(y.shape) > 1:
        raise Exception("To many dimensions to plot!")


    # rearrange in ascending order
    x = np.array(x)
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    # now copy from vis. and also get labels as add. parameter
    ax.plot(x, y, label=label)


def reconstruct_experiment(path):
    snapshots = load_data(path)

    nodes = sorted(list(snapshots.keys()))
    iterations_nos = sorted(list(snapshots[nodes[0]].keys()))
    n_mins = [len(t[0]) for t in snapshots[nodes[0]][iterations_nos[0]].explanations]

    data = [[[[t[2], t[1]] for t in snapshots[n][i].explanations] for i in iterations_nos] for n in nodes]
    data = np.array(data)

    return data, nodes, iterations_nos, n_mins

def display_experiments(paths):
    fig, ax = plt.subplots()

    path = paths[0]
    data, nodes, iterations_nos, n_mins = reconstruct_experiment(path)
    for i in iterations_nos:
        plot_experiment(ax, data, nodes, iterations_nos, n_mins, label=i,
                        n=None, m=i, n_min='x')
    ax.legend()
    ax.set_xlabel('Sparsity')
    ax.set_ylabel('Fidelity')
    plt.show()

    fig, ax = plt.subplots()

    for size in n_mins:
        plot_experiment(ax, data, nodes, iterations_nos, n_mins, label=f'{size} nodes',
                        n=None, m='x', n_min=size)
    ax.legend()
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Fidelity')
    plt.show()

def display_experiment_old(path):
    data, nodes, iterations_nos, n_mins = reconstruct_experiment(path)

    sparsity_list = []
    fidelity_list = []
    labels = []

    for i in iterations_nos:
        sx_dict = {n: [t + (0,) for t in snapshots[n][i].explanations] for n in snapshots}

        #  sx_dict = {n: [] for n in {s.index for s in snapshots}}
        #  for s in snapshots:
        #      sx_dict[s.index].append(s.explanations)

        sx_sparsity, sx_fidelity = aggregate_fidelity_sparsity(sx_dict)
        sparsity_list.append(sx_sparsity)
        fidelity_list.append(sx_fidelity)
        labels.append(f'{i}')

    plot_results(sparsity_list, fidelity_list, labels, save_dst='./img/karate_club/karate_experiments_first.png')

def prepare_experiment():
    dataset = torch_geometric.datasets.KarateClub()
    graph = dataset.data
    device = get_device()

    emb_model = karate_club.train_or_load_embedding(graph, only_load=True)
    graph, train_loader, val_loader, test_loader = karate_club.prepare_dataset(emb_model)

    model, loss_func = karate_club.train_or_load_gcn(train_loader, val_loader, only_load=True)
    test_loss, test_acc = test(model, False, test_loader, loss_func, task=Task.NODE_CLASSIFICATION)
    print(f'test loss: {test_loss}, test_acc: {test_acc}')

    sx_params = {
        "model": model,
        "num_layers": 2,
        "exp_weight": 5,
        "m": 30,
        "t": 50,
        "task": Task.NODE_CLASSIFICATION,
        "max_children": 12,
    }

    # experiment_params = {
    #     "snapshot_after": [1, 5, 10, 15],
    #     "n_mins": [4, 5, 6, 7, 8, 9, 10, 11, 12],
    # }

    return sx_params, test_loader


def run_experiment(path, sx_params, graph_loader):
    set_seed(0)  # IMPORTANT!!
    snapshot_after = [1, 5, 10]  #, 15]  #, 20, 30]
    n_mins = [4, 5, 6, 7, 8, 9, 10, 11, 12]

    # nodes of test set
    test_graph = next(iter(graph_loader))
    test_mask = test_graph.test_mask
    test_idx = torch.arange(0, len(test_mask))[test_mask].tolist()
    test_idx = [1, 3]  # WE WANT SOME RESULTS, FAST

    # load old snapshots and add new ones
    if os.path.isfile(path):
        print('Warning: going to overwrite old snapshots.')
        snapshots = load_data(path)
    else:
        snapshots = dict()
        for node in test_idx:
            snapshots[node] = dict()
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

    save_data(path, snapshots)


def main():
    path = './result_data/karate_club/experiments/first'
    # sx_params, graph_loader = prepare_experiment()
    # run_experiment(path, sx_params, graph_loader)
    display_experiments([path])

if __name__ == '__main__':
    set_seed(0)  # IMPORTANT!!
    main()
