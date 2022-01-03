from typing import List, Union

from matplotlib import pyplot as plt

def plot_results(sparsity_list: List[List[float]], fidelity_list: List[List[float]], labels: List[str],
                 save_dst: Union[None, str] = None):
    for sparsity, fidelity, label in zip(sparsity_list, fidelity_list, labels):
        plt.plot(sparsity, fidelity, label=label)
    plt.legend()
    plt.xlabel('Sparsity')
    plt.ylabel('Fidelity')

    if save_dst is not None:
        plt.savefig(save_dst)

    plt.show()
