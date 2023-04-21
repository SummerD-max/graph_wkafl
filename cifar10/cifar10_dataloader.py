import logging

import syft as sy
import torch

from models import DGI
from utils import aug
from utils import process
import scipy.sparse as sp

logger = logging.getLogger(__name__)


class testDataLoader():
    def __init__(self, data, targets):
        self.data = data
        self.labels = targets

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def testLoader(testset):
    datas = testset.data
    data = [torch.unsqueeze(1. * torch.tensor(datas[i].transpose()), 0) for i in range(10000)]
    data = torch.cat(data, 0)
    return testDataLoader(data, testset.targets)


def get_federated_graph_dataset(dataset_name, n_clients):
    """
    distribute dataset to all the clients
    :param dataset_name: the name of the dataset
    :param n_clients: num of clients
    """

    # Store the training data and labels of all the clients
    datasets = []
    # TODO: populate this datasets list with ((adj, feature..), labels) in the following for loop

    for i in range(n_clients):
        # path of features
        path_feat = "/home/amax/lym/SAFA_semiAsyncFL-master/data/{}/{}/{}_{}_feat.txt".format(
            dataset_name, n_clients, dataset_name, i
        )
        # path of edges
        path_edge = "/home/amax/lym/SAFA_semiAsyncFL-master/data/{}/{}/{}_{}.txt".format(
            dataset_name, n_clients, dataset_name, i
        )

        adj, features, seq = process.load_data_2(path_edge, path_feat)
        num_nodes = features.shape[0]
        ft_size = features.shape[1]

        print(f"第{i}个参与方的节点数量:{num_nodes}")
        print(f"第{i}个参与方的特征数量:{ft_size}")

        # hyperparameter
        hid_units = 256
        activation_fc = 'relu'
        lr = 0.001
        weight_decay = 0.0001

        # define model and optimizer
        model = DGI(n_in=ft_size, n_h=hid_units, activation=activation_fc)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

        # define augmentation option
        aug_type = 'mask'
        drop_percent = 0.4

        # data augmentation
        if aug_type == 'edge':
            aug_features1 = features
            aug_features2 = features
            aug_adj1 = aug.aug_random_edge(adj, drop_percent=drop_percent)
            aug_adj2 = aug.aug_random_edge(adj, drop_percent=drop_percent)
        elif aug_type == 'subgraph':
            aug_features1, aug_adj1 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
            aug_features2, aug_adj2 = aug.aug_subgraph(features, adj, drop_percent=drop_percent)
        elif aug_type == 'node':
            aug_features1, aug_adj1 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
            aug_features2, aug_adj2 = aug.aug_drop_node(features, adj, drop_percent=drop_percent)
        elif aug_type == 'mask':  # Change the node features, preserving the graph topology
            aug_features1 = aug.aug_random_mask(features, drop_percent=drop_percent)
            aug_features2 = aug.aug_random_mask(features, drop_percent=drop_percent)
            aug_adj1 = adj
            aug_adj2 = adj
        else:
            raise ValueError("the aug_type is incorrect")

        # normalize adj
        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))

        # normalize aug_adj
        aug_adj1 = process.normalize_adj(aug_adj1 + sp.eye(aug_adj1.shape[0]))
        aug_adj2 = process.normalize_adj(aug_adj2 + sp.eye(aug_adj2.shape[0]))

        # sparse adj
        sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)

        # sparse aug_adj
        sp_aug_adj1 = process.sparse_mx_to_torch_sparse_tensor(aug_adj1)
        sp_aug_adj2 = process.sparse_mx_to_torch_sparse_tensor(aug_adj2)

        # specify device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("using gpu")

            # move training model to device
            model.to(device)

            # move features to device
            features = features.to(device)

            # move aug_features to device
            aug_features1 = aug_features1.to(device)
            aug_features2 = aug_features2.to(device)

            # move sp_adj to device
            sp_adj = sp_adj.to(device)

            # move sp_aug_adj to device
            sp_aug_adj1 = sp_aug_adj1.to(device)
            sp_aug_adj2 = sp_aug_adj2.to(device)

            # train_data = (features, shuffle_features, aug_features1, aug_features2, sp_adj, sp_aug_adj1, sp_aug_adj2)

        # TODO: populate datasets list with ((adj, feature..), labels)
        # for example: datasets.append(sy.frameworks.torch.fl.dataset.BaseDataset(user_data, user_label))
        # user_data: Tuple(adj, feature)
        # user_label: [0, 1]???
        datasets.append()

def dataset_federate_noniid(trainset, workers, classNum):
    """
    Add a method to easily transform a torch.Dataset or a sy.BaseDataset
    into a sy.FederatedDataset. The dataset given is split in len(workers)
    part and sent to each workers
    """
    logger.info(f"Scanning and sending data to {', '.join([w.id for w in workers])}...")
    datas = trainset.data
    labels = trainset.targets
    labels = torch.tensor(labels)
    dataset = {}

    data_new = []
    for i in range(50000):
        data_new.append(torch.unsqueeze(1. * torch.tensor(datas[i].transpose()), 0))
    datas = torch.cat(data_new, 0)

    for i in range(10):
        index = (labels == i)
        dataset[str(i)] = datas[index]

    datasets = []
    datasTotalNum = []
    user_num = len(workers)

    for i in range(user_num):
        user_data = []
        user_label = []
        labelClass = torch.randperm(10)[0:classNum]
        dataRate = torch.rand([classNum])
        dataRate = dataRate / torch.sum(dataRate)
        dataNum = torch.randperm(40)[0] + 10
        dataNum = torch.round(dataNum * dataRate)
        if classNum > 1:
            datasnum = torch.zeros([10])
            datasnum[labelClass.tolist()] = dataNum
            datasTotalNum.append(datasnum)
            for j in range(classNum):
                datanum = int(dataNum[j].item())
                index = torch.randperm(5000)[0:datanum]
                user_data.append(dataset[str(labelClass[j].item())][index, :, :, :])
                user_label.append(labelClass[j] * torch.ones(datanum))
            user_data = torch.cat(user_data, 0)
            user_label = torch.cat(user_label, 0)

        else:
            j = 0
            datasnum = torch.zeros([10])
            datasnum[labelClass] = dataNum
            datasTotalNum.append(datasnum)

            datanum = int(dataNum[j].item())
            index = torch.randperm(5000)[0:datanum]
            user_data = dataset[str(labelClass[j].item())][index, :, :, :]
            user_label = labelClass[j] * torch.ones(datanum)
            user_data = torch.tensor(user_data)

        worker = workers[i]
        logger.debug("Sending data to worker %s", worker.id)
        user_data = user_data.send(worker)
        user_label = user_label.send(worker)
        datasets.append(sy.frameworks.torch.fl.dataset.BaseDataset(user_data, user_label))
        # datasets.append(sy.BaseDataset(user_data, user_label))  # .send(worker)
    logger.debug("Done!")
    return sy.FederatedDataset(datasets), datasTotalNum


'''
import torchvision as tv            #里面含有许多数据集
import torch
import torchvision.transforms as transforms    #实现图片变换处理的包
from torchvision.transforms import ToPILImage

#使用torchvision加载并预处理CIFAR10数据集
show = ToPILImage()         #可以把Tensor转成Image,方便进行可视化
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean = (0.5,0.5,0.5),std = (0.5,0.5,0.5))])#把数据变为tensor并且归一化range [0, 255] -> [0.0,1.0]
trainset = tv.datasets.CIFAR100(root='data1/', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)
testset = tv.datasets.CIFAR100('data1/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True, num_workers=0)

(data,label) = trainset[100]
show((data+1)/2).resize((100, 100))
dataiter = iter(trainloader)
images, labels = dataiter.next()
#print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid((images+1)/2)).resize((400, 100))#make_grid的作用是将若干幅图像拼成一幅图像

#定义网络
import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,6,5)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,100)
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return  x

net = Net()
print(net)

# 定义损失函数和优化器
from torch import optim
criterion = nn.CrossEntropyLoss()    # 定义交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

# 训练网络
from torch.autograd import Variable
for epoch in range(1):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):    # enumerate将其组成一个索引序列，利用它可以同时获得索引和值,enumerate还可以接收第二个参数，用于指定索引起始值
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)

        _, predicted = torch.max(outputs.data, 1)
        total = labels.size(0)
        correct = (predicted == labels).sum()

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 ==0:
            print('[{}, {}] loss: {}, acc is {}'.format(epoch+1,i+1,running_loss/2000, 1.*correct/total))
            running_loss = 0.0
print("----------finished training---------")
dataiter = iter(testloader)
images, labels = dataiter.next()
# print('实际的label: ',' '.join('%08s'%classes[labels[j]] for j in range(4)))
show(tv.utils.make_grid(images/2 - 0.5)).resize((400,100))
outputs = net(Variable(images))
_, predicted = torch.max(outputs.data, 1)    # 返回最大值和其索引
# print('预测结果:',' '.join('%5s'%classes[predicted[j]] for j in range(4)))
correct = 0
total = 0
for data in testloader:
    images, labels = data
    outputs = net(Variable(images))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
print('10000张测试集中的准确率为: %d %%'%(100*correct/total))
if torch.cuda.is_available():
    net.cuda()
    images = images.cuda()
    labels = labels.cuda()
    output = net(Variable(images))
    loss = criterion(output, Variable(labels))
'''
