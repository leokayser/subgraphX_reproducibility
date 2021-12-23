import math
from collections import defaultdict

from typing import Set, Callable, List, Dict, Union

import networkx as nx
import torch.nn
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


class MCTSNode:
    def __init__(self, graph: Data, n_min: int, node_set: Set[int]):
        self.graph = graph
        self.n_min = n_min
        self.node_set = node_set

    def is_terminal(self) -> bool:
        return len(self.node_set) <= self.n_min

    def __hash__(self) -> int:
        return hash(list(self.node_set).sort())

    def get_pruned_nodes(self) -> Set[int]:  # inverse of node set
        return set(range(self.graph.num_nodes)) - self.node_set

    def __str__(self) -> str:
        return f'{sorted(list(self.get_pruned_nodes()))}'

    def __eq__(self, node2) -> bool:
        return self.node_set == node2.node_set


class MCTS:
    def __init__(self, graph: Data, exp_weight: float, n_min: int, score_func: Callable, model: torch.nn.Module,
                 t: int, num_layers: int, single_output: bool):
        self.W = defaultdict(float)  # total reward of each node
        self.C = defaultdict(int)  # total visit count for each node
        self.children: Dict[MCTSNode, List[MCTSNode]] = {}  # nodes and their children
        self.leaves = []  # all terminal nodes
        self.R: Dict[MCTSNode, float] = {}  # immediate reward for nodes

        self.exp_weight = exp_weight  # lambda
        self.n_min = n_min
        self.score_func = score_func
        self.single_output = single_output
        self.graph = graph
        self.model = model
        self.t = t
        self.num_layers = num_layers

        self.root = MCTSNode(graph, n_min, set(range(graph.num_nodes)))

    def _q(self, mcts_node) -> float:
        if self.C[mcts_node] == 0:
            return 0  # avoid unseen moves
        return self.W[mcts_node] / self.C[mcts_node]  # average reward

    def _r(self, mcts_node) -> float:
        if mcts_node in self.R.keys():
            return self.R[mcts_node]
        else:
            score = self.score_func(self.model, self.graph, mcts_node.node_set, self.t, self.num_layers,
                                    self.single_output)
            self.R[mcts_node] = score
            return score

    def _u(self, mcts_node, parent) -> float:  # utility from paper
        children = self.children[parent]
        counts = [self.C[n] for n in children]
        parent_count = sum(counts)
        return self.exp_weight * self._r(mcts_node) * math.sqrt(parent_count) / (1 + self.C[mcts_node])

    def _ucb(self, node, parent) -> float:  # upper confidence bound
        return self._q(node) + self._u(node, parent)

    def _select_path_by_ucb(self) -> List[MCTSNode]:  # choose best leaf by ucb, training
        mcts_node = self.root
        path = [mcts_node]
        while not mcts_node.is_terminal():
            mcts_node = self._best_child_by_ucb(mcts_node)
            path.append(mcts_node)
        return path

    def _best_child_by_ucb(self, mcts_node: MCTSNode):
        def _score_helper(child):
            return self._ucb(child, mcts_node)

        if mcts_node in self.children.keys():
            children = self.children[mcts_node]
        else:
            children = self._expand_node(mcts_node)

        return max(children, key=_score_helper)

    def _expand_node(self, mcts_node) -> List[MCTSNode]:
        if mcts_node in self.children.keys():
            raise Exception(f'Node is already expanded: {mcts_node}')

        if mcts_node.is_terminal():
            self.children[mcts_node] = []
            return []

        children = []
        nx_graph = to_networkx(self.graph, to_undirected=True)  # connected components only works for directed graphs

        for node in mcts_node.node_set:
            subgraph = nx_graph.subgraph(mcts_node.node_set - {node})
            components = nx.connected_components(subgraph)
            child_set = max(components, key=lambda x: len(x))  # only keep largest connected component

            child = MCTSNode(self.graph, self.n_min, child_set)
            children.append(child)

        self.children[mcts_node] = children  # TODO: sort children according to strategy
        return children

    def _backpropagate(self, path):
        score = self._r(path[-1])  # score is reward of leaf node
        for mcts_node in path:
            self.C[mcts_node] += 1
            self.W[mcts_node] += score
        pass  # put debug breakpoint here, very useful

    def search_one_iteration(self):  # train for one iteration
        path = self._select_path_by_ucb()
        leaf = path[-1]
        if leaf not in self.leaves:
            self.leaves.append(leaf)
        self._backpropagate(path)

    def best_leaf_node(self) -> MCTSNode:  # choose best leaf by reward only
        return max(self.leaves, key=self._r)

    def _node_info(self, mcts_node) -> str:
        pruned_nodes = list(mcts_node.get_pruned_nodes())
        pruned_nodes.sort()
        if mcts_node in self.children.keys():
            children_list = [str(child) for child in self.children[mcts_node]]
        else:
            children_list = []
        return f'-Removed {pruned_nodes}: {self._r(mcts_node):.6f}, children:{children_list}'

    def print_tree_sequential(self):  # mainly for debugging purposes
        print(f'mcts search tree for {self.graph}')
        print(f'Number of nodes in search tree:  {len(self.W)}')

        i = 0
        for mcts_node in self.W.keys():
            print(f'{i}: {mcts_node.info()}')
            i += 1
