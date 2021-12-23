# Generic Code for training and Evaluating Neural Networks
from typing import Callable, Tuple, Union, List

import torch
from torch import optim
from torch_geometric.loader import DataLoader

from src.utils.utils import get_device


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    correct = (pred == target)
    num_correct = correct.sum().item()
    num_items = correct.shape[0]
    return num_correct / num_items


def _episode_helper(model: torch.nn.Module, single_output: bool, optimizer: Union[optim.Optimizer, None],
                    loader: DataLoader, loss_func: Callable, train: bool) -> Tuple[float, float]:
    device = get_device()
    model = model.to(device)

    loss = 0
    acc = 0

    for graph in iter(loader):
        if train:
            optimizer.zero_grad()

        scores = model(graph.x.to(device), graph.edge_index.to(device), graph.batch.to(device))
        if single_output:
            scores = scores.squeeze()
        cur_loss = loss_func(scores, graph.y.to(device))

        if train:
            cur_loss.backward()
            optimizer.step()

        if single_output:
            pred = torch.round(scores).detach().cpu()
        else:
            pred = torch.argmax(scores, dim=1).detach().cpu()
        acc += compute_accuracy(pred, graph.y)
        loss += cur_loss.detach().cpu().item()

    loss /= len(loader)
    acc /= len(loader)

    return loss, acc


def _episode(model: torch.nn.Module, single_output: bool, optimizer: Union[optim.Optimizer, None], loader: DataLoader,
             loss_func: Callable, train: bool) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    if not train:
        with torch.no_grad():
            return _episode_helper(model, single_output, optimizer, loader, loss_func, train)
    else:
        return _episode_helper(model, single_output, optimizer, loader, loss_func, train)


def test(model: torch.nn.Module, single_output: bool, test_loader: DataLoader, loss_func: Callable) \
        -> Tuple[float, float]:
    loss, acc = _episode(model, single_output, None, test_loader, loss_func, train=False)
    return loss, acc


def train_model(model: torch.nn.Module, single_output: bool, optimizer: optim.Optimizer, train_loader: DataLoader,
                val_loader: DataLoader, num_epochs: int, loss_func: Callable, save_dst: str, output_freq: int = 1) \
                -> Tuple[List[float], List[float], List[float], List[float]]:
    best_epoch = -1
    best_val_acc = 0

    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    # initial acc
    init_loss, init_acc = _episode(model, single_output, None, val_loader, loss_func, train=False)
    print(f'Initial Val Accuracy: {init_acc:.4f}, initial Loss: {init_loss}')

    for i in range(1, num_epochs + 1):

        # training
        cur_train_loss, cur_train_acc = _episode(model, single_output, optimizer, train_loader, loss_func, train=True)
        train_loss.append(cur_train_loss)
        train_acc.append(cur_train_acc)

        # eval
        cur_val_loss, cur_val_acc = _episode(model, single_output, None, val_loader, loss_func, train=False)
        val_loss.append(cur_val_loss)
        val_acc.append(cur_val_acc)
        if cur_val_acc > best_val_acc:
            best_val_acc = cur_val_acc
            best_epoch = i
            torch.save(model.state_dict(), save_dst)  # save the best model

        # output
        if (i % output_freq) == 0:
            print(f'Epoch: {i}, \ttrain_loss: {train_loss[-1]:.4f} \tacc_train: {train_acc[-1]:.4f}'
                  + f' \tval_loss: {val_loss[-1]:.4f} \tval_acc: {val_acc[-1]:.4f}')

    print(f'Finished Training. Best Val Acc in epoch {best_epoch}: {best_val_acc:.4f}, Loss: '
          + f'{val_loss[best_epoch - 1]}')
    model.load_state_dict(torch.load(save_dst))  # load the best model

    return train_loss, train_acc, val_loss, val_acc
