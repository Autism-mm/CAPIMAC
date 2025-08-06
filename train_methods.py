import numpy as np

from model import SdA
from config import *
import torch.nn as nn
import torch
import time
import logging
import torch.nn.functional as F
def train1(train_pairs, model, criterion, optimizer, epoch, args):
    if epoch % 10 == 0:
        logging.info("=======> Train epoch: {}/{}".format(epoch, args.epochs))
    model.train()
    time0 = time.time()
    loss_value = 0
    x0,x1=torch.from_numpy(train_pairs[0]).float(),torch.from_numpy(train_pairs[1]).float()
    x0, x1 = x0.to(args.gpu), x1.to(args.gpu)
    # print(np.shape(x0))
    try:
        h0, h1, d0, d1 = model(x0, x1)
    except:
        print("error raise in batch",epoch)
    #
    # x0, x1 = torch.squeeze(x0), torch.squeeze(x1)
    loss = criterion(x0, d0)
    loss += criterion(x1, d1)
    loss += model.regularization_loss()#l2正则化
    loss_value += loss.item()
    if epoch != 0:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_time = time.time() - time0

    return h0 , h1,epoch_time
def pretrain(train_pairs, args):
    model = SdA(config).to(args.gpu)
    criterion = nn.MSELoss().to(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    # 'train'
    for i in range(0, args.epochs + 1):
        if i == 0:
            with torch.no_grad():
                h0, h1, epoch_time = train1(train_pairs, model, criterion, optimizer, i, args)
        else:
            h0, h1, epoch_time = train1(train_pairs, model, criterion, optimizer, i, args)
    return h0, h1, epoch_time

def train2(train_loader, model, criterion,optimizer, epoch, args):
    model.train()
    time0 = time.time()
    loss_value = 0
    for batch_idx, (x0, x1, labels, real_labels) in enumerate(train_loader):
        # labels refer to noisy labels for the constructed pairs, while real_labels are the clean labels for these pairs
        x0, x1, labels, real_labels = x0.to(args.gpu), x1.to(args.gpu), labels.to(args.gpu), real_labels.to(args.gpu)
        print(np.shape(x0))
        try:
            h0, h1 = model(x0.view(x0.size()[0], -1), x1.view(x1.size()[0], -1))
        except:
            print("error raise in batch", batch_idx)

        pair_dist = F.pairwise_distance(h0, h1)

        loss = criterion(pair_dist, labels, args.margin, args.robust, args)
        # loss1=criterion_mse(z0, z1)
        # print(loss1,'loss')
        loss_value += loss.item()
        if epoch != 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    epoch_time = time.time() - time0
    return epoch_time