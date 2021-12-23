from typing import List, Set, Union, Tuple

import numpy as np
import torch
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from src.utils.utils import get_device


@torch.no_grad()
def _aggregate_scores(loader, model, class_idx, single_output: bool) -> torch.Tensor:
    device = get_device()
    result = torch.tensor([]).float()
    for data in iter(loader):
        scores = model(data.x.to(device), data.edge_index.to(device), data.batch.to(device)).detach().cpu()
        if single_output:
            score_of_class = scores.squeeze()
        else:
            score_of_class = scores[:, class_idx]

        result = torch.cat((result, score_of_class), dim=0)
    return result


@torch.no_grad()
def _compute_marginal_contribution(include_list, exclude_list, model, class_idx, single_output: bool) -> float:
    include_loader = DataLoader(include_list, batch_size=64, shuffle=False, num_workers=0)
    exclude_loader = DataLoader(exclude_list, batch_size=64, shuffle=False, num_workers=0)

    include_scores = _aggregate_scores(include_loader, model, class_idx, single_output)
    exclude_scores = _aggregate_scores(exclude_loader, model, class_idx, single_output)

    contribution = torch.mean(include_scores - exclude_scores).item()  # TODO: this is technically wrong for single_output and class 0
    return contribution


@torch.no_grad()
def mc_l_shapley(model: torch.nn.Module, graph: Data, subgraph: Set[int], t: int, num_layers: int,
                 single_output: bool) -> float:
    """
    Shapley computation by monte carlo approximation in local neighborhood
    :param model:
    :param graph:
    :param subgraph:
    :param t:
    :param num_layers:
    :param single_output:
    :return:
    """
    device = get_device()
    # initialize coalition space
    subgraph_list = list(subgraph)  # v1 to vk
    node_tensor, edge_index, mapping, _ = k_hop_subgraph(subgraph_list, num_layers, graph.edge_index,
                                                         relabel_nodes=False, num_nodes=graph.num_nodes,
                                                         flow='source_to_target')  # source to target is important
    reachable_list = node_tensor.tolist()  # v1 to vr
    p_prime = list(set(reachable_list) - set(subgraph_list))

    placeholder = graph.num_nodes
    p = p_prime + [placeholder]

    # assuming graph classification
    scores = model(x=graph.x.to(device), edge_index=graph.edge_index.to(device),
                   batch=torch.zeros(graph.num_nodes).long().to(device)).detach().cpu()
    if len(scores.shape) > 1:
        scores = scores.squeeze()

    if single_output:
        predicted_class = int(torch.round(scores).item())
    else:
        predicted_class = torch.argmax(scores, dim=0).item()

    exclude_data_list = []
    include_data_list = []

    for i in range(t):
        perm = np.random.permutation(p)
        split_idx = np.asarray(perm == placeholder).nonzero()[0][0]

        selected = perm[:split_idx]  # nodes selected for coalition

        include_mask = np.zeros(graph.num_nodes)
        include_mask[selected] = 1
        include_mask[subgraph_list] = 1
        masked_x_include = graph.x * torch.tensor(include_mask).unsqueeze(1)  # zero padding
        include_data = Data(masked_x_include.float(), graph.edge_index)
        include_data_list.append(include_data)

        exclude_mask = np.zeros(graph.num_nodes)
        exclude_mask[selected] = 1
        masked_x_exclude = graph.x * torch.tensor(exclude_mask).unsqueeze(1)  # zero padding
        exclude_data = Data(masked_x_exclude.float(), graph.edge_index)
        exclude_data_list.append(exclude_data)

    score = _compute_marginal_contribution(include_data_list, exclude_data_list, model, predicted_class, single_output)
    return score
