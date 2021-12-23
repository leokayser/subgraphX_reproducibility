from typing import Set

import torch
from torch_geometric.data import Data

from src.utils.utils import get_predicted_class, get_scores


def fidelity(graph: Data, node_set: Set[int], model: torch.nn.Module, single_output: bool) -> float:
    node_tensor = torch.tensor(list(node_set))

    x_occluded = torch.clone(graph.x)
    x_occluded[node_tensor] = 0

    batch = torch.zeros(graph.num_nodes).long()

    predicted_class = get_predicted_class(model, graph.x, graph.edge_index, batch, single_output)

    scores = get_scores(model, graph.x, graph.edge_index, batch, train=False)
    scores_occluded = get_scores(model, x_occluded, graph.edge_index, batch, train=False)

    if single_output:
        result = (scores - scores_occluded).item()  # TODO: this is technically wrong for single_output and class 0
    else:
        scores = scores[:, predicted_class]
        scores_occluded = scores_occluded[:, predicted_class]
        result = (scores - scores_occluded).item()

    return result


def sparsity(graph: Data, node_set: Set[int]) -> float:
    return 1 - (len(node_set) / graph.num_nodes)


