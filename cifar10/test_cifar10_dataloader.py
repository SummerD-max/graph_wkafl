from cifar10.cifar10_dataloader import get_federated_graph_dataset

def test_get_federated_graph_dataset():
    get_federated_graph_dataset('citeseer', 50)