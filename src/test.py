from dig.xgraph.dataset import SynGraphDataset
from dig.xgraph.models import *
import torch
from dig.xgraph.method import SubgraphX

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def main():
    dataset = SynGraphDataset('./datasets', 'BA_shapes')
    dataset.data.x = dataset.data.x.to(torch.float32)
    dataset.data.x = dataset.data.x[:, :1]
    dim_node = dataset.num_node_features
    dim_edge = dataset.num_edge_features
    num_classes = dataset.num_classes

    model = GCN_2l(model_level='node', dim_node=dim_node, dim_hidden=300, num_classes=num_classes)
    explainer = SubgraphX(model, num_classes=4, device=device, explain_graph=False, reward_method='nc_mc_l_shapley')


if __name__ == '__main__':
    main()