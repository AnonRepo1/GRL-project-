from diffgin import DiffGIN
from mincut import MinCutGIN
from dmon import DMonGIN
from edge import Edge_Pool_GIN
from base import Base_GIN_Net
from topk import Top_K_Pool_Net
import torch
import numpy as np
import torch.nn.functional as F
from math import ceil
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import to_dense_adj, to_dense_batch
import csv
import time

rng = np.random.default_rng(1)
# Load the MUTAG dataset
model_names = ["dmon" ]
dataset_names = ["Mutagenicity"]
def data(name):
    dataset = TUDataset(root=f'/tmp/{name}', name=name)
    dataset = dataset.shuffle()
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    avg_nodes = int(dataset.data.num_nodes/len(dataset))
    max_nodes = 0
    for d in dataset:
        max_nodes = max(d.num_nodes, max_nodes)

    return dataset, num_features, num_classes, avg_nodes, max_nodes

def NetModel(model_name, num_features, num_classes, avg_nodes, max_nodes, pool_ratio=0.1):
    if model_name == "diff":
        return DiffGIN(in_channels=num_features, out_channels=num_classes,pool_ratio=pool_ratio,average_nodes=avg_nodes).to(device)
    elif model_name == "mincut":
        return MinCutGIN(in_channels=num_features, out_channels=num_classes,pool_ratio=pool_ratio,average_nodes=avg_nodes).to(device)
    elif model_name == "dmon":
        return DMonGIN(in_channels=num_features, out_channels=num_classes,pool_ratio=pool_ratio,average_nodes=avg_nodes).to(device)
    elif model_name == "edge":
        return Edge_Pool_GIN(in_channels=num_features, out_channels=num_classes,pool_ratio=pool_ratio).to(device)
    elif model_name == "base":
        return Base_GIN_Net(in_channels=num_features, out_channels=num_classes).to(device)
    elif model_name == "topk":
        return Top_K_Pool_Net(in_channels=num_features, out_channels=num_classes).to(device)
    

"""def __init__(self,
                 in_channels,
                 out_channels,
                 num_layers_pre=1,
                 num_layers_post=1,
                 hidden_channels=64,
                 norm=True,
                 activation="ELU",
                 average_nodes=None,
                 max_nodes=None,
                 pooling=None,
                 pool_ratio=0.1):"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def log(**kwargs):
    def _map(value) -> str:
        if isinstance(value, int) and not isinstance(value, bool):
            return f'{value:03d}'
        if isinstance(value, float):
            return f'{value:.4f}'
        return value

    print(', '.join(f'{key}: {_map(value)}' for key, value in kwargs.items()))
def train(model,loader, optimizer):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out, aux_loss = model(data)
        loss = F.nll_loss(out, data.y) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += float(loss)*data.num_graphs
    return total_loss/len(loader.dataset)

def test(model, loader):
    model.eval()
    total_correct = 0
    total_loss = 0
    for data in loader:
        data = data.to(device)
        out, _ = model(data)
        loss = F.nll_loss(out, data.y)
        total_loss += float(loss)*data.num_graphs
        pred = out.argmax(dim=-1)
        total_correct += int((pred==data.y).sum())
    return total_correct/len(loader.dataset), total_loss/len(loader.dataset)

def train_and_evaluate(model_name="diff_pool", dataset_name="MUTAG", runs=1, epochs=100, pool_ratio=0.1):
    dataset, num_features, num_classes, avg_nodes, max_nodes = data(dataset_name)

    tot_acc = []
    epoch_times = []
    for r in range(runs):

        rnd_idx = rng.permutation(len(dataset))
        dataset = dataset[list(rnd_idx)]

        train_dataset = dataset[len(dataset) // 5:]
        val_dataset = dataset[: len(dataset)// 10]
        test_dataset = dataset[len(dataset)//10: len(dataset)//5]

        train_loader = DataLoader(train_dataset, batch_size=32,shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32)
        test_loader = DataLoader(test_dataset, batch_size=32)

        net_model = NetModel(model_name, num_features, num_classes, avg_nodes, max_nodes)
        opt = torch.optim.Adam(net_model.parameters(), lr=1e-4)

        #Train
        best_val=np.inf
        best_test=0
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            loss = train(net_model, train_loader, opt)
            train_acc, _ = test(net_model, train_loader)
            val_acc, val_loss = test(net_model, val_loader)
            test_acc, _ = test(net_model, test_loader)
            end_time = time.time()  # End time of epoch
            epoch_duration = end_time - start_time  # Duration of epoch
            epoch_times.append(epoch_duration)  # Append duration to list

            if val_loss < best_val:
                best_val = val_loss
                best_test = test_acc
            log(Epoch=epoch, Loss=loss, Train=train_acc, Val=val_acc, Test=test_acc, Time=epoch_duration)
            
        tot_acc.append(best_test)
        average_time_per_epoch = sum(epoch_times) / len(epoch_times)
        print(f"### Run {r:d} - val loss: {best_val:.3f}, test acc: {best_test:.3f}")

    print("Accuracies in each run: ", tot_acc)    
    print(f"test acc - mean: {np.mean(tot_acc):.3f}, std: {np.std(tot_acc):.3f}, {model_name}, {pool_ratio}")

    return np.mean(tot_acc), np.std(tot_acc), average_time_per_epoch

csv_file = open(f"dmon.csv", mode='w', newline='', encoding='utf-8')
field_names = ["model","dataset", "mean accuracy", "standard deviation", "average time per epoch"]
writer = csv.DictWriter(csv_file, fieldnames=field_names)
writer.writeheader()
for name in model_names:
    for tudset in dataset_names:
        tot_acc, std, avg_time = train_and_evaluate(model_name=name,dataset_name=tudset)
        writer.writerow({
            'model': name,
            'dataset': tudset,
            'mean accuracy': tot_acc,
            'standard deviation': std,
            "average time per epoch": avg_time
        })

csv_file.close()

