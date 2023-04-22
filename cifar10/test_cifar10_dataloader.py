import pytest
import syft as sy
import torch
from cifar10.cifar10_dataloader import get_federated_graph_dataset

@pytest.fixture
def user_num():
    return 50

@pytest.fixture
def workers(user_num):
    hook = sy.TorchHook(torch)
    workers = []
    for i in range(1, user_num + 1):
        exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
        exec('workers.append(user{})'.format(i))

    return workers
def test_get_federated_graph_dataset(workers):
    clients_data_dict, dataset = get_federated_graph_dataset('citeseer', workers)
    print(clients_data_dict)

