from typing import Set

import torch
from torch_geometric.data import Data

from src.utils.task_enum import Task
from src.utils.utils import get_predicted_class, get_scores


def fidelity(graph: Data, node_set: Set[int], model: torch.nn.Module, task: Task = Task.GRAPH_CLASSIFICATION,
             index_of_interest: int = -1) -> float:
    """
    Compute fidelity of an explanation
    :param graph: Graph instance to explain
    :param node_set: Explanation in form of a node set
    :param model: model to be explained
    :param task: graph, node classification or link prediction
    :param index_of_interest: node or link to be explained. Not necessary for graph prediction
    :return: Fidelity score
    """
    if task == Task.NODE_CLASSIFICATION:  # exclude node to explain from zero padding
        node_set = node_set - {index_of_interest}

    node_tensor = torch.tensor(list(node_set))

    x_occluded = torch.clone(graph.x)
    x_occluded[node_tensor] = 0

    batch = torch.zeros(graph.num_nodes).long()

    scores = get_scores(model, graph.x, graph.edge_index, batch, train=False)
    scores_occluded = get_scores(model, x_occluded, graph.edge_index, batch, train=False)
    if task == Task.NODE_CLASSIFICATION:
        scores = scores[index_of_interest]
        scores_occluded = scores_occluded[index_of_interest]

    # predicted_class = get_predicted_class(model, graph.x, graph.edge_index, batch, single_output=False)
    if task == Task.GRAPH_CLASSIFICATION:
        predicted_class = torch.argmax(scores, dim=1)
        scores = scores[:, predicted_class]
        scores_occluded = scores_occluded[:, predicted_class]
        result = (scores - scores_occluded).item()

    elif task == Task.NODE_CLASSIFICATION:
        predicted_class = torch.argmax(scores, dim=0)
        scores = scores[predicted_class]
        scores_occluded = scores_occluded[predicted_class]
        result = (scores - scores_occluded).item()

    else:
        raise NotImplementedError('link')

    return result


def sparsity(graph: Data, node_set: Set[int]) -> float:
    return 1 - (len(node_set) / graph.num_nodes)


