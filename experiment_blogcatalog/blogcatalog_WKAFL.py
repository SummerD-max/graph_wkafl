import logging
import time
from datetime import datetime

import syft as sy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from cifar10.cifar10_dataloader import get_federated_graph_dataset
from models.utils import get_DGI_model_optimizer

from models.dgi import DGI

hook = sy.TorchHook(torch)
logger = logging.getLogger(__name__)
date = datetime.now().strftime('%Y-%m-%d %H:%M')


class Argument():
    def __init__(self):
        # self.user_num = 2000     # number of total clients P
        # 实验二场景二
        self.user_num = 50
        # self.K = 20     # number of participant clients K
        self.K = 25

        self.CB1 = 70  # clip parameter in both stages
        self.CB2 = 5  # clip parameter B at stage two
        self.lr = 0.0005  # learning rate of global model
        self.itr_test = 100  # number of iterations for the two neighbour tests on test datasets
        self.batch_size = 8  # batch size of each client for local training
        self.test_batch_size = 128  # batch size for test datasets
        self.total_iterations = 50  # total number of iterations
        self.stageTwo = 3500  # the iteration of stage one
        # self.total_iterations =200
        # self.stageTwo = 100
        self.threshold = 0.3  # threshold to judge whether gradients are consistent
        self.classNum = 2  # the number of data classes for each client
        self.alpha = 0.1  # parameter for momentum to alleviate the effect of non-IID data
        self.seed = 1  # parameter for the server to initialize the model
        self.cuda_use = False


args = Argument()
use_cuda = args.cuda_use and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "gpu")


# 定义网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()  # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
#         self.conv1 = nn.Conv2d(3, 6, 5)  # 定义conv1函数的是图像卷积函数：输入为图像（3个频道，即RGB图）,输出为 6张特征图, 卷积核为5x5正方形
#         self.conv2 = nn.Conv2d(6, 16, 5)  # 定义conv2函数的是图像卷积函数：输入为6张特征图,输出为16张特征图, 卷积核为5x5正方形
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将16*5*5个节点连接到120个节点上。
#         self.fc2 = nn.Linear(120, 84)  # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将120个节点连接到84个节点上。
#         self.fc3 = nn.Linear(84, 10)  # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将84个节点连接到10个节点上。
#
#     # 定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # 输入x经过卷积conv1之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 输入x经过卷积conv2之后，经过激活函数ReLU，使用2x2的窗口进行最大池化Max pooling，然后更新到x。
#         x = x.view(-1, self.num_flat_features(x))  # view函数将张量x变形成一维的向量形式，总特征数并不改变，为接下来的全连接作准备。
#         x = F.relu(self.fc1(x))  # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
#         x = F.relu(self.fc2(x))  # 输入x经过全连接2，再经过ReLU激活函数，然后更新x
#         x = self.fc3(x)  # 输入x经过全连接3，然后更新x
#         return x


##################################获取模型层数和各层的形状#############
def GetModelLayers(model):
    Layers_shape = []
    Layers_nodes = []
    for idx, params in enumerate(model.parameters()):
        Layers_num = idx
        Layers_shape.append(params.shape)
        Layers_nodes.append(params.numel())
    return Layers_num + 1, Layers_shape, Layers_nodes


##################################设置各层的梯度为0#####################
def ZerosGradients(Layers_shape) -> list:
    """

    Args:
        Layers_shape:

    Returns:
        ZoroGradient: model parameters in 0
    """
    ZeroGradient = []
    for i in range(len(Layers_shape)):
        ZeroGradient.append(torch.zeros(Layers_shape[i]).to(device))
    return ZeroGradient


################################调整学习率###############################
def lr_adjust(args, tau):
    """

    Args:
        args: arguments
        tau: the minimum of taus among current K clients

    Returns:

    """
    tau = 0.01 * tau + 1
    lr = args.lr / tau
    return lr


#################################计算范数################################
def L_norm(Tensor):
    norm_Tensor = torch.tensor([0.])
    for i in range(len(Tensor)):
        norm_Tensor += Tensor[i].float().norm() ** 2
    return norm_Tensor.sqrt()


################################# 计算角相似度 ############################
def similarity(user_Gradients, yun_Gradients):
    sim = torch.tensor([0.])
    for i in range(len(user_Gradients)):
        sim = sim + torch.sum(user_Gradients[i] * yun_Gradients[i])
    if L_norm(user_Gradients) == 0:
        print('梯度为0.')
        sim = torch.tensor([1.])
        return sim
    sim = sim / (L_norm(user_Gradients) * L_norm(yun_Gradients))
    return sim


#################################聚合####################################
def aggregation(Collect_Gradients, K_Gradients, weight, Layers_shape, args, Clip=False, Layers_num=None):
    sim = torch.zeros([args.K])
    Gradients_Total = torch.zeros([args.K + 1])
    for i in range(args.K):
        Gradients_Total[i] = L_norm(K_Gradients[i])
    Gradients_Total[args.K] = L_norm(Collect_Gradients)
    # print('Gradients_norm', Gradients_Total)
    for i in range(args.K):
        sim[i] = similarity(K_Gradients[i], Collect_Gradients)
    index = (sim > args.threshold)
    # print('sim:', sim)
    if sum(index) == 0:
        print("相似度均较低")
        return Collect_Gradients
    Collect_Gradients = ZerosGradients(Layers_shape)

    totalSim = []
    Sel_Gradients = []
    for i in range(args.K):
        if sim[i] > args.threshold:
            totalSim.append((torch.exp(sim[i] * 50) * weight[i]).tolist())
            Sel_Gradients.append(K_Gradients[i])
    totalSim = torch.tensor(totalSim)
    totalSim = totalSim / torch.sum(totalSim)
    for i in range(len(totalSim)):
        Gradients_Sample = Sel_Gradients[i]
        if Clip:
            # standNorm = Gradients_Total[len(Gradients_Total)]
            standNorm = L_norm(Collect_Gradients)
            Gradients_Sample = TensorClip(Gradients_Sample, args.CB2 * standNorm, Layers_num)
        for j in range(len(K_Gradients[i])):
            Collect_Gradients[j] += Gradients_Sample[j] * totalSim[i]
    return Collect_Gradients


################################ 定义剪裁 #################################
def TensorClip(Tensor, ClipBound, Layers_num):
    norm_Tensor = L_norm(Tensor)
    if ClipBound < norm_Tensor:
        for i in range(Layers_num):
            Tensor[i] = Tensor[i] * ClipBound / norm_Tensor
    return Tensor


############################定义测试函数################################
def test(model, testx_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    with torch.no_grad():
        for imagedata, labels in test_loader:
            outputs = model(Variable(imagedata))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_acc = 100. * correct / total
    end = time.time()
    print('Running time: %s Seconds' % (end - start))
    print('10000张测试集: testacc is {:.4f}%, testloss is {}.'.format(test_acc, test_loss))
    return test_loss, test_acc


##########################定义训练过程，返回梯度########################
def train(model, train_data, train_targets, optimizer, global_model=None):
    # start training
    model.train()
    # unpack
    features_tensor, shuffle_features, aug_features1, aug_features2, sp_adj, sp_aug_adj1, sp_aug_adj2 = train_data

    # move training data to gpu
    features_tensor = features_tensor.to(device)
    shuffle_features = shuffle_features.to(device)
    aug_features1 = aug_features1.to(device)
    aug_features2 = aug_features2.to(device)
    sp_adj = sp_adj.to(device)
    sp_aug_adj1 = sp_aug_adj1.to(device)
    sp_aug_adj2 = sp_aug_adj2.to(device)

    # move labels to gpu
    train_targets = train_targets.to(device)

    # start training
    model.train()
    model.zero_grad()
    optimizer.zero_grad()

    # forward prop
    aug_type = "node"
    logits = model(features_tensor,
                   shuffle_features,
                   aug_features1,
                   aug_features2,
                   sp_adj,
                   sp_aug_adj1,
                   sp_aug_adj2,
                   True, None, None, None, aug_type=aug_type)

    train_targets = train_targets.unsqueeze(0)
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(logits, train_targets)
    loss.backward()
    optimizer.step()

    Gradients_Tensor = []
    for params in model.parameters():
        Gradients_Tensor.append(params.grad.data)  # 把各层的梯度添加到张量Gradients_Tensor
    return Gradients_Tensor, loss


##################################计算Non-IID程度##################################
def JaDis(datasNum, userNum):
    sim = []
    for i in range(userNum):
        data1 = datasNum[i]
        for j in range(i + 1, userNum):
            data2 = datasNum[j]
            sameNum = [min(x, y) for x, y in zip(data1, data2)]
            sim.append(sum(sameNum) / (sum(data1) + sum(data2) - sum(sameNum)))
    distance = 2 * sum(sim) / (userNum * (userNum - 1))
    return distance


def main():
    # 模型和用户生成
    start = time.time()
    dataset_name = "pubmed"
    feat_path = f"/home/amax/lym/SAFA_semiAsyncFL-master/data/{dataset_name}/50/{dataset_name}_0_feat.txt"
    with open(feat_path, 'r') as f:
        for line in f:
            line_list = line.strip('\n').split(' ')
            feat_size = len(line_list) - 1
            break

    global_model, optimizer = get_DGI_model_optimizer(feat_size)  # global model and optimizer

    # 获取模型层数和各层形状
    Layers_num, Layers_shape, Layers_nodes = GetModelLayers(global_model)
    e = torch.exp(torch.tensor(1.))

    workers = []
    models = {}
    optims = {}
    taus = {}

    # define user1-user{user_num}, and push them to workers list, initialize taus to 1
    for i in range(1, args.user_num + 1):
        exec('user{} = sy.VirtualWorker(hook, id="user{}")'.format(i, i))
        exec('workers.append(user{})'.format(i))  # 列表形式存储用户
        exec('taus["user{}"] = {}'.format(i, 1))  # 列表形式存储用户
        # exec('workers["user{}"] = user{}'.format(i,i))    # 字典形式存储用户

    # distribute initial models and optimizers
    for i in range(1, args.user_num + 1):
        exec('models["user{}"] = global_model.copy()'.format(i))
        exec('optims["user{}"] = optimizer'.format(i, i))
    optim_sever = optim.SGD(params=global_model.parameters(), lr=args.lr)  # 定义服务器优化器

    # 数据载入
    # clients_data_dict: graph data and labels
    clients_data_dict, federated_datasets = get_federated_graph_dataset("BlogCataLog", workers)

    # =======================================================
    # TODO: how to deal with testset??
    # testset = tv.datasets.CIFAR10('data2/', train=False, download=True, transform=transform)
    # testset = cifar10_dataloader.testLoader(testset)
    # test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=True, drop_last=True,
    #                                           num_workers=0)

    # 定义记录字典
    # logs = {'train_loss': [], 'test_loss': [], 'test_acc': []}
    # test_loss, test_acc = test(model, test_loader, device)  # 初始模型的预测精度
    # logs['test_acc'].append(test_acc.item())
    # logs['test_loss'].append(test_loss)
    #

    # ===========================================================

    # 联邦学习过程

    for key in models:
        models[key] = models[key].to(device)

    # 定义训练/测试过程
    for itr in range(1, args.total_iterations + 1):
        # 按设定的每回合用户数量和每个用户的批数量载入数据，单个批的大小为batch_size
        # 为了对每个样本上的梯度进行裁剪，令batch_size=1，batch_num=args.batch_size*args.batchs_round，将样本逐个计算梯度
        federated_train_loader = sy.FederatedDataLoader(federated_datasets,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        worker_num=args.K,
                                                        batch_num=1)
        workers_list = federated_train_loader.workers  # 当前回合抽取的用户列表

        # 生成与模型梯度结构相同的元素=0的列表
        Loss_train = torch.tensor(0.)
        weight = []
        K_tau = []
        K_Gradients = []
        for idx_outer, (worker_id, worker_id) in enumerate(federated_train_loader):
            client_id = str(worker_id.location.id)
            model_round = models[client_id]
            optimizer = optims[client_id]
            K_tau.append(taus[client_id])

            train_data, train_targets = clients_data_dict[client_id]

            # optimizer = optims[data.location.id]
            # 返回梯度张量，列表形式；同时返回loss；gradient=False，则返回-lr*grad

            # TODO: 在这里改模型model_round(该客户端的训练模型)、训练数据train_data改为图数据、优化器改成GCN所用到的优化器，
            print(f"training_idx: {idx_outer}, training client: {client_id}")
            Gradients_Sample, loss = train(model_round,
                                           train_data,
                                           train_targets,
                                           optimizer,
                                           global_model=global_model)
            Loss_train += loss
            if itr > 1:
                for j in range(Layers_num):
                    Gradients_Sample[j] = Gradients_Sample[j] + args.alpha * collect_gradients[j]
            K_Gradients.append(TensorClip(Gradients_Sample, args.CB1, Layers_num))

        collect_gradients = ZerosGradients(Layers_shape)
        for i in range(len(collect_gradients)):
            collect_gradients[i] = collect_gradients[i].to(device)

        K_tau = torch.tensor(K_tau) * 1.
        _, index = torch.sort(K_tau)
        normStandard = L_norm(K_Gradients[index[0]])
        weight = (e / 2) ** (-K_tau)
        if torch.sum(weight) == 0:
            print("延时过大。")
            for i in range(Layers_num):
                weight[index[0]] = 1.
                collect_gradients = K_Gradients[index[0]]
        else:
            weight = weight / torch.sum(weight)
            for i in range(args.K):
                Gradients_Sample = K_Gradients[i]
                for j in range(Layers_num):
                    collect_gradients[j] += Gradients_Sample[j] * weight[i]

        if itr < args.stageTwo:
            collect_gradients = aggregation(collect_gradients, K_Gradients, weight, Layers_shape, args,
                                            Layers_num=Layers_num)
        elif itr > 100:
            collect_gradients = aggregation(collect_gradients, K_Gradients, weight, Layers_shape, args, Clip=True,
                                            Layers_num=Layers_num)

        # 升级延时信息
        for tau in taus:
            taus[tau] = taus[tau] + 1
        for worker in workers_list:
            taus[worker] = 1

        lr = lr_adjust(args, torch.min(K_tau))
        for grad_idx, params_sever in enumerate(global_model.parameters()):
            params_sever.data.add_(-lr, collect_gradients[grad_idx])

        # 同步更新不需要下面代码；异步更新需要下段代码
        # global model -> client model (last )
        for worker_idx in range(len(workers_list)):
            worker_model = models[workers_list[worker_idx]]
            for idx, (params_server, params_client) in enumerate(
                    zip(global_model.parameters(), worker_model.parameters())):
                params_client.data = params_server.data
            models[workers_list[worker_idx]] = worker_model  ###添加把更新后的模型返回给用户

        # TODO: 这里精度实验以后再补上去
        # if itr == 1 or itr % args.itr_test == 0:
        #     print('itr: {}'.format(itr))
        #     test_loss, test_acc = test(global_model, test_loader, device)  # 初始模型的预测精度
        #     logs['test_acc'].append(test_acc.item())
        #     logs['test_loss'].append(test_loss)
        #     logs['train_loss'].append(Loss_train.item())
    ##########写入文件############

    # with open('./results/cifar10_WKAFL_testacc.txt', 'a+') as fl:
    #     fl.write('\n' + date + '%Results (UN is {}, K is {}, threshold is {}, CB1 is {}, CB2 is {}, BZ is {}, LR is {}, '.
    #              format(args.user_num, args.K, args.threshold, args.CB1, args.CB2, args.batch_size, args.lr))
    #     fl.write('total itr is {}, itr_test is {}, stageTwo is {}, classNum is {})\n'.
    #              format(args.total_iterations, args.itr_test, args.stageTwo, args.classNum))
    #     fl.write('WKAFL: ' + str(logs['test_acc']))
    #
    # with open('./results/cifar10_WKAFL_testloss.txt', 'a+') as fl:
    #     fl.write('\n' + date + ' Results (UN is {}, K is {}, threshold is {}, CB1 is {}, CB2 is {}, BZ is {}, LR is {}, '.
    #              format(args.user_num, args.K, args.threshold, args.CB1, args.CB2, args.batch_size, args.lr))
    #     fl.write('total itr is {}, itr_test is {}, stageTwo is {}, classNum is {})\n'.
    #              format(args.total_iterations, args.itr_test, args.stageTwo, args.classNum))
    #     fl.write('testloss: ' + str(logs['test_loss']))
    #
    # with open('./results/cifar10_WKAFL_trainloss.txt', 'a+') as fl:
    #     fl.write('\n' + date + ' Results (UN is {}, K is {}, threshold is {}, CB1 is {}, CB2 is {}, BZ is {}, LR is {}, '.
    #              format(args.user_num, args.K, args.threshold, args.CB1, args.CB2, args.batch_size, args.lr))
    #     fl.write('total itr is {}, itr_test is {}, stageTwo is {}, classNum is {})\n'.
    #              format(args.total_iterations, args.itr_test, args.stageTwo, args.classNum))
    #     fl.write('trainloss: ' + str(logs['train_loss']))

    ##########################
    end = time.time()
    print('Running time: %s Seconds' % (end - start))

    # TODO: 这里将测试集精度和训练集损失记录在txt文件中
    # with open('/home/amax/lpy/cifar10/cifar10_WKAFL_testacc_1.txt', 'a+') as fl:
    #     fl.write(
    #         '\n' + date + '%Results (UN is {}, K is {}, threshold is {}, CB1 is {}, CB2 is {}, BZ is {}, LR is {}, '.
    #         format(args.user_num, args.K, args.threshold, args.CB1, args.CB2, args.batch_size, args.lr))
    #     fl.write('total itr is {}, itr_test is {}, stageTwo is {}, classNum is {})\n'.
    #              format(args.total_iterations, args.itr_test, args.stageTwo, args.classNum))
    #     fl.write('WKAFL: ' + str(logs['test_acc']))
    #
    # with open('/home/amax/lpy/cifar10/cifar10_WKAFL_testloss_1.txt', 'a+') as fl:
    #     fl.write(
    #         '\n' + date + ' Results (UN is {}, K is {}, threshold is {}, CB1 is {}, CB2 is {}, BZ is {}, LR is {}, '.
    #         format(args.user_num, args.K, args.threshold, args.CB1, args.CB2, args.batch_size, args.lr))
    #     fl.write('total itr is {}, itr_test is {}, stageTwo is {}, classNum is {})\n'.
    #              format(args.total_iterations, args.itr_test, args.stageTwo, args.classNum))
    #     fl.write('testloss: ' + str(logs['test_loss']))
    #
    # with open('/home/amax/lpy/cifar10/cifar10_WKAFL_trainloss_1.txt', 'a+') as fl:
    #     fl.write(
    #         '\n' + date + ' Results (UN is {}, K is {}, threshold is {}, CB1 is {}, CB2 is {}, BZ is {}, LR is {}, '.
    #         format(args.user_num, args.K, args.threshold, args.CB1, args.CB2, args.batch_size, args.lr))
    #     fl.write('total itr is {}, itr_test is {}, stageTwo is {}, classNum is {})\n'.
    #              format(args.total_iterations, args.itr_test, args.stageTwo, args.classNum))
    #     fl.write('trainloss: ' + str(logs['train_loss']))
    # =====================================================================================


if __name__ == "__main__":
    main()
