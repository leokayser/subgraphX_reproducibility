import random
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# reproducibility
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch_geometric.seed_everything(seed)


def get_scores(model: torch.nn.Module, x: torch.tensor, edge_index: torch.tensor, batch: torch.tensor,
               train: bool) -> torch.tensor:
    device = get_device()
    model.to(device)

    if train:
        scores = model(x.to(device), edge_index.to(device), batch.to(device)).detach().cpu()
    else:
        with torch.no_grad():
            scores = model(x.to(device), edge_index.to(device), batch.to(device)).detach().cpu()
    return scores


def get_predicted_class(model: torch.nn.Module, x: torch.tensor, edge_index: torch.tensor,
                        batch: torch.tensor, single_output: bool) -> torch.tensor:
    scores = get_scores(model, x, edge_index, batch, train=False)

    if single_output:
        scores = scores.squeeze()
        return torch.round(scores)
    else:
        return torch.argmax(scores, dim=1)

