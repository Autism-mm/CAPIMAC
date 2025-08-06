import argparse
import time
import random
from model import *
import math
import torch,gc
import torch.nn as nn
import torch.nn.functional as F
from train_methods import *
import logging
import sys
import numpy as np
import matplotlib.pyplot as plt
from Datasets import *
from config import *
from data_loader import *
import mat73
from anchors import *
from Cluster import *
parser = argparse.ArgumentParser(description='CAPIMAC in PyTorch')
parser.add_argument('--data', default='1', type=int,
                    help='choice of dataset, 0-HW,1-3Sources,2BBC,3-Scene15, 4-Caltech101,5-ORL_mtv,6-Caltech_7,7-Reuters,'
                         '8-20newsgroups,9-100leaves,10-BBC4,11-MSRCv1,12-BDGP,13-HandWritten,14-yale_mtv，15-Wikipedia-test,16-Movies,17-Prokaryotic,18-ALOI,19-flower17')
parser.add_argument('-bs', '--batch-size', default='1024', type=int, help='number of batch size')
parser.add_argument('-e', '--epochs', default='200', type=int, help='number of epochs to run')
parser.add_argument('-lr', '--learn-rate', default='0.0001', type=float, help='learning rate of adam')
parser.add_argument('-ap', '--aligned-prop', default='0.5', type=float,
                    help='originally aligned proportions in the partially view-aligned data')
parser.add_argument('--gpu', default=0, type=int, help='GPU device idx to use.')
parser.add_argument('-cp', '--complete-prop', default='0.5', type=float,
                    help='originally complete proportions in the partially sample-missing data')
parser.add_argument('-m', '--margin', default='5', type=int, help='initial margin')
parser.add_argument('-s', '--start-fine', default=True, type=bool, help='flag to start use robust loss or not')
parser.add_argument('-np', '--neg-num', default='30', type=int, help='the ratio of negative to positive pairs')
parser.add_argument('-noise', '--noisy-training', type=bool, default=True,
                    help='training with real labels or noisy labels')
parser.add_argument('-r', '--robust', default=1, type=int, help='use our robust loss or not')

dim=0
class NoiseRobustLoss(nn.Module):
    def __init__(self):
        super(NoiseRobustLoss, self).__init__()

    def forward(self, pair_dist, P, margin, use_robust_loss, args):
        # print(max(pair_dist))
        dist_sq = pair_dist * pair_dist
        P = P.to(torch.float32)
        N = len(P)
        if use_robust_loss == 1:
            if args.start_fine:
                loss = P * dist_sq + (1 - P) * (1 / margin) * torch.pow(
                    torch.clamp(torch.pow(pair_dist, 0.5) * (0.5*margin - pair_dist), min=0.0), 2)
            else:
                loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        else:
            loss = P * dist_sq + (1 - P) * torch.pow(torch.clamp(margin - pair_dist, min=0.0), 2)
        loss = torch.sum(loss) / (2.0 * N)
        return loss
def load_data(align_prop,complete_prop,neg_num,is_noise,dataset):
    global dim
    NetSeed = random.randint(1, 1000)
    # NetSeed=72
    print(NetSeed)
    np.random.seed(NetSeed)
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(NetSeed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(NetSeed)  # 为当前GPU设置随机种子
    args = parser.parse_args()
    all_data = []
    map_pairs = []
    label = []
    train_pairs = []

    if dataset=='Caltech101_7':
        path = './datasets/' + dataset + '.mat'  # 路径
        mat = mat73.loadmat(path)  # 加载mat文件
    else:
        mat = sio.loadmat('./datasets/' + dataset + '.mat')
    if dataset == 'Scene15':
        data = mat['X'][0][0:2]  # 20, 59 dimensions
        label = np.squeeze(mat['Y'])
    elif dataset == 'HandWritten':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])
    elif dataset == '3Sources':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'ALOI':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'BBCsports':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Caltech101':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == 'Reuters_dim10':
        data = []  # 18758 samples
        data.append(normalize(np.vstack((mat['x_train'][0], mat['x_test'][0]))))
        data.append(normalize(np.vstack((mat['x_train'][1], mat['x_test'][1]))))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))
    elif dataset == 'ORL_mtv':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'Caltech101_7':
        data = mat['data'][3:5]
        data[0], data[1] = np.squeeze(data[0]), np.squeeze(data[1])
        data[0], data[1] = np.array(data[0]), np.array(data[1])
        label = np.squeeze(mat['labels'])
    elif dataset == 'Reuters':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    elif dataset == '20NewsGroups':
        data = mat['data'][0][1:3]
        label = np.squeeze(mat['truelabel'][0][0])
    elif dataset == '100leaves':
        mat['data'][0][0], mat['data'][0][1] = mat['data'][0][0].T, mat['data'][0][1].T
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])
    elif dataset == 'BBC4':
        data = mat['data'][0][0:2]
        label = np.squeeze(mat['truelabel'][0][0])
        # print(label)
    elif dataset == 'MSRCv1':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])
    elif dataset == 'BDGP':
        mat['X'][0][0], mat['X'][0][1] = mat['X'][0][0].T, mat['X'][0][1].T
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['gt'])
    elif dataset == 'HandWritten':
        data = mat['X'][0][1:3]
        label = np.squeeze(mat['Y'])
    elif dataset == 'yale_mtv':
        mat['X'][0][0], mat['X'][0][1] = mat['X'][0][0].T, mat['X'][0][1].T
        data = mat['X'][0][0:2]
        # print((data))
        label = np.squeeze(mat['gt'])
    elif dataset == 'Wikipedia-test':
        data = mat['X'][0:2][0:2]
        data = np.squeeze(data.T)
        # print(data)
        label = np.squeeze(mat['y'])
    elif dataset == 'Movies':
        data = mat['X'][0:2][0:2]
        data = np.squeeze(data.T)
        # print(data)
        label = np.squeeze(mat['y'])
    elif dataset == 'Prokaryotic':
        value1 = mat['X'][0][0]
        value2 = mat['X'][2][0]
        data = [value1, value2]
        # print(data)
        label = np.squeeze(mat['y'])
    elif dataset == 'flower17':
        data = mat['X'][0][0:2]
        label = np.squeeze(mat['Y'])
    divide_seed = random.randint(1, 1000)
    train_idx, test_idx = TT_split(len(label), 1 - align_prop, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx]
    if dataset == 'Caltech101_7':
        data[0], data[1] = np.squeeze(data[0]), np.squeeze(data[1])
    print(np.shape(data[0]))
    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx], data[0][test_idx], data[1][test_idx]
    '''获取对齐部分的潜在表示'''
    map_pairs.append(train_X)
    map_pairs.append(train_Y)
    h0 , h1,epoch_time=pretrain(map_pairs, args)
    all_label = np.concatenate((train_label, test_label))
    '''获取初始训练数据和测试数据'''
    if align_prop != 1:
        shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))
        test_Y = test_Y[shuffle_idx]
        test_label_X, test_label_Y = test_label, test_label[shuffle_idx]
    elif align_prop == 1:
        all_data.append(train_X.T)
        all_data.append(train_Y.T)
    '''不完整部分'''
    test_mask = get_sn(2, len(test_label), 1 - complete_prop)
    X_mask, Y_mask = test_mask[:, 0].astype(np.bool_), test_mask[:, 1].astype(np.bool_)
    # test_X[~X_mask] = 0
    # test_Y[~Y_mask] = 0
    test_X, test_Y = test_X[X_mask], test_Y[Y_mask]
    test_label_X, test_label_Y=test_label_X[X_mask], test_label_Y[Y_mask]
    if align_prop != 1:
        all_label_X = np.concatenate((train_label, test_label_X))
        all_label_Y = np.concatenate((train_label, test_label_Y))
        all_data.append(np.concatenate((train_X, test_X)).T)
        all_data.append(np.concatenate((train_Y, test_Y)).T)
        all_label = np.concatenate((train_label, test_label))
        # all_label_X = test_label_X
        # all_label_Y = test_label_Y
        # all_data.append(test_X.T)
        # all_data.append(test_Y.T)
        # all_label = test_label
    elif align_prop == 1:
        all_label_X, all_label_Y = train_label, train_label
        all_label = train_label
    '''构建训练对'''
    view0, view1, noisy_labels, real_labels, _, _ = get_pairs(train_X, train_Y, neg_num, train_label)
    count = 0
    for i in range(len(noisy_labels)):
        if noisy_labels[i] != real_labels[i]:
            count += 1
    print('noise rate of the constructed neg. pairs is ', round(count / (len(noisy_labels) - len(train_X)), 2))

    if is_noise == 0:  # training with real_labels, v/t with real_labels
        print("----------------------Training with real_labels----------------------")
        train_pair_labels = real_labels
    else:  # training with labels, v/t with real_labels
        print("----------------------Training with noisy_labels----------------------")
        train_pair_labels = noisy_labels
    '''初始化锚点'''
    num_unique_labels = np.unique(all_label).shape[0]

    anchors0,anchors1,len_indices=get_anchors(h0,h1,map_pairs,num_unique_labels)#h0是tensor

    '''数据重表示'''
    view0,view1,all_data[0],all_data[1]=torch.from_numpy(view0).float(),torch.from_numpy(view1).float(),torch.from_numpy(all_data[0]).float(),torch.from_numpy(all_data[1]).float()

    view0, view1, all_data[0],all_data[1]=find_nanchor(anchors0,view0),find_nanchor(anchors1,view1),find_nanchor(anchors0,all_data[0].T),find_nanchor(anchors1,all_data[1].T)
    #锚点数×样本数,增强锚点图
    view0, view1, all_data[0], all_data[1]=np.array(view0),np.array(view1),np.array(all_data[0]),np.array(all_data[1])
    print(np.shape(view0),'view0')
    train_pairs.append(view0)
    train_pairs.append(view1)
    train_pair_real_labels = real_labels
    dim=view0.shape[0]
    return train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, dim,num_unique_labels,divide_seed

def normalize(x):
    x = (x - np.tile(np.min(x, axis=0), (x.shape[0], 1))) / np.tile((np.max(x, axis=0) - np.min(x, axis=0)),
                                                                    (x.shape[0], 1))
    return x
def loader(train_bs, align_prop, complete_prop,neg_num, is_noise, dataset):
    """
    :param train_bs: batch size for training, default is 1024
    :param neg_prop: negative / positive pairs' ratio
    :param test_prop: known aligned proportions for training MvCLN
    :param is_noise: training with noisy labels or not, 0 --- not, 1 --- yes
    :param data_idx: choice of dataset
    :return: train_pair_loader including the constructed pos. and neg. pairs used for training MvCLN, all_loader including originally aligned and unaligned data used for testing MvCLN
    """
    train_pairs, train_pair_labels, train_pair_real_labels, all_data, all_label, all_label_X, all_label_Y, dim,num_unique_labels,divide_seed\
        = load_data(align_prop,complete_prop,neg_num,is_noise, dataset)
    train_pair_dataset = GetDataset(train_pairs, train_pair_labels, train_pair_real_labels)

    train_pair_loader = DataLoader(
        train_pair_dataset,
        batch_size=train_bs,
        shuffle=True,
        drop_last=True
    )
    return train_pair_loader, all_data, all_label, all_label_X, all_label_Y, dim,num_unique_labels,divide_seed

if __name__ == '__main__':
    for i in range(1):
        args = parser.parse_args()
        data_name = ['HandWritten', '3Sources', 'BBCsports', 'Scene15', 'Caltech101', 'ORL_mtv', 'Caltech101_7', 'Reuters',
                 '20NewsGroups','100leaves','BBC4','MSRCv1','BDGP','HandWritten','yale_mtv','Wikipedia-test','Movies','Prokaryotic','ALOI','flower17']
        train_pair_loader, all_data, all_label, all_label_X, all_label_Y, dim, outfeature ,divide_seed=loader(args.batch_size, args.aligned_prop,args.complete_prop,args.neg_num,args.noisy_training,data_name[args.data])

        model = Anchormodel(dim,outfeature).to(args.gpu)
        criterion = NoiseRobustLoss().to(args.gpu)
        # criterion_mse = nn.MSELoss().to(args.gpu)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
        CAR_list = []
        acc_list, nmi_list, ari_list,f_list,f1_list,pre_list,pre2_list,rec_list,pur_list = [], [], [],[], [], [],[], [], []
        train_time = 0
        all_data[0], all_data[1]=torch.from_numpy(all_data[0]), torch.from_numpy(all_data[1])
        for i in range(0, args.epochs + 1):
            if i == 0:
                with torch.no_grad():
                    epoch_time = train2(train_pair_loader, model, criterion, optimizer, i, args)
            else:
                epoch_time = train2(train_pair_loader, model, criterion, optimizer, i, args)
            # test
            v0, v1, pred_label, alignment_rate = tiny_infer(model, args.gpu, all_data, all_label_X, all_label_Y)
            CAR_list.append(alignment_rate)
            data = []
            data.append(v0)
            data.append(v1)

            y_pred, ret, accuracy, nmi, ari, f_score, f_score2, precision, precision2, recall, purity = Clustering(data,
                                                                                                                   pred_label)
            if i % 10 == 0:
                print(accuracy, nmi, ari, f_score, f_score2, precision, precision2, recall, purity)
                # logging.info("******** testing ********")
                # logging.info(
                #     "CAR={} kmeans: acc={} nmi={} ari={}".format(round(alignment_rate, 4), ret['kmeans']['accuracy'],
                #                                                  ret['kmeans']['NMI'], ret['kmeans']['ARI']))
            acc_list.append(ret['kmeans']['ACC'])
            nmi_list.append(ret['kmeans']['NMI'])
            ari_list.append(ret['kmeans']['ARI'])
            f_list.append(ret['kmeans']['F1'])
            f1_list.append(ret['kmeans']['F2'])
            pre_list.append(ret['kmeans']['PRE'])
            pre2_list.append(ret['kmeans']['PRE2'])
            rec_list.append(ret['kmeans']['REC'])
            pur_list.append(ret['kmeans']['PUR'])
        print('ACC:', max(acc_list))
        print("NMI:", max(nmi_list))
        print("ARI:", max(ari_list))
        print("F1:", max(f_list))
        print("F2:", max(f1_list))
        print("PRE:", max(pre_list))
        print("PRE2:", max(pre2_list))
        print("REC:", max(rec_list))
        print("PUR:", max(pur_list))
        logging.info('******** End, training time = {} s ********'.format(round(train_time, 2)))