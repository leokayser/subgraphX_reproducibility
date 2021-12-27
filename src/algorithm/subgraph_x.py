import torch.nn
from torch_geometric.data import Data

from src.algorithm.mcts import MCTS
from src.algorithm.shapley import mc_l_shapley


class SubgraphX:
    def __init__(self, model: torch.nn.Module, num_layers: int, exp_weight: float, m: int, t: int):
        self.model = model
        self.num_layers = num_layers
        self.exp_weight = exp_weight
        self.m = m
        self.t = t
        self.value_func = mc_l_shapley  # only implementation for now

    def _get_mcts(self, graph: Data, n_min: int):
        return MCTS(graph, self.exp_weight, n_min, self.value_func, self.model, self.t,
                    self.num_layers)

    def __call__(self, graph: Data, n_min: int):
        # graph classification
        mcts = self._get_mcts(graph, n_min)

        for iteration in range(self.m):
            mcts.search_one_iteration()

        explanation = mcts.best_leaf_node()

        return explanation.node_set, mcts


