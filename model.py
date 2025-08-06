import torch.nn as nn
from torch import optim

class endA(nn.Module):
    def __init__(self, in_features, out_features):
        super(endA, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(True),
            nn.Dropout(0.1)
        )  # 编码

        #self.decoder[0].weight.data = self.encoder[0].weight.data.transpose(0, 1)

    def forward(self, x):
        h = self.encoder(x)
        return h


class dedA(nn.Module):
    def __init__(self, out_features, in_features):
        super(dedA, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(out_features, in_features),
            nn.ReLU(True)
        )  # 编码

        # self.decoder[0].weight.data = self.encoder[0].weight.data.transpose(0, 1)

    def forward(self, x):
        h = self.decoder(x)
        return h

class SdA(nn.Module):
    def __init__(self, config):
        super(SdA, self).__init__()

        layers1 = []
        layers2 = []
        layersall1=[]
        layersall2 = []
        in_features1 = config.input_features1


        for out_features in config.enhidden_features:
            layer1 = endA(in_features1, out_features)
            in_features1 = out_features
            layers1.append(layer1)

        self.layers1 = nn.Sequential(*layers1)  # 就是封装了成了一个

        in_features=config.enhidden_features[-1]
        for out_features in config.dehidden_features1:
            layer2 = dedA(in_features, out_features)
            in_features = out_features
            layers2.append(layer2)

        self.layers2=nn.Sequential(*layers2)

        layersall1.append(self.layers1)
        layersall1.append(self.layers2)
        self.layerll1=nn.Sequential(*layersall1)

        if config.is_train:
            self.ce_criterion = nn.CrossEntropyLoss()
            self.da_optimizers = []
            for layer1 in self.layers1[:-1]:
                # optimizer = optim.SGD(layer1.parameters(), lr=config.lr,
                #                       momentum=config.momentum, weight_decay=config.weight_decay)  # 优化器可以改一下
                optimizer = optim.Adam(
                    layer1.parameters(), lr=0.001, betas=(0.9, 0.99), eps=1e-8, weight_decay=0)
                self.da_optimizers.append(optimizer)

        layers3 = []
        layers4 = []
        in_features2 = config.input_features2
        for out_features in config.enhidden_features:
            layer3 = endA(in_features2, out_features)
            in_features2 = out_features
            layers3.append(layer3)

        self.layers3 = nn.Sequential(*layers3)  # 就是封装了成了一个

        in_features=config.enhidden_features[-1]
        for out_features in config.dehidden_features2:
            layer4 = dedA(in_features, out_features)
            in_features = out_features
            layers4.append(layer4)

        self.layers4=nn.Sequential(*layers4)

        layersall2.append(self.layers3)
        layersall2.append(self.layers4)
        self.layerll2 = nn.Sequential(*layersall2)
        # for layer in self.layers3:
        #     print(layer)

        if config.is_train:
            self.ce_criterion = nn.CrossEntropyLoss()
            self.da_optimizers = []
            for layer1 in self.layers3[:-1]:
                # optimizer = optim.SGD(layer1.parameters(), lr=config.lr,
                #                       momentum=config.momentum, weight_decay=config.weight_decay)  # 优化器可以改一下
                optimizer=optim.Adam(layer1.parameters(),lr=0.001,betas=(0.9,0.99),eps=1e-8,weight_decay=0)
                self.da_optimizers.append(optimizer)
            # 每一层的优化器


    def forward(self, x1, x2):
        h1, h2 = x1, x2
        for layer1 in self.layers1:
            h1 = layer1(h1)
        h3 = h1
        for layer2 in self.layers2:
            h3 = layer2(h3)
        for layer3 in self.layers3:
            h2 = layer3(h2)
        h4=h2
        for layer4 in self.layers4:
            h4 = layer4(h4)
        return h1, h2, h3, h4  # 不是很理解构

    def regularization_loss(self):
        l2_lambda = 0.001
        l2_norm = sum(p.pow(2).sum() for p in self.parameters())
        return l2_lambda * l2_norm


class Anchormodel(nn.Module):
    def __init__(self,dim,outfeature):
        super(Anchormodel, self).__init__()
        self.encoder0 = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, outfeature),
            nn.BatchNorm1d(outfeature),
            nn.ReLU(True)
        )
        self.encoder1 = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(1024, outfeature),
            nn.BatchNorm1d(outfeature),
            nn.ReLU(True)
        )

        # self.decoder0 = nn.Sequential(nn.Linear(outfeature, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
        #                               nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
        #                               nn.Linear(1024, dim))
        # self.decoder1 = nn.Sequential(nn.Linear(outfeature, 1024), nn.ReLU(), nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(),
        #                               nn.Dropout(0.2), nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.2),
        #                               nn.Linear(1024, dim))
    def forward(self, x0, x1):
        h0 = self.encoder0(x0.view(x0.size()[0], -1))
        h1 = self.encoder1(x1.view(x1.size()[0], -1))
        # z0 = self.decoder0(h0)
        # z1 = self.decoder1(h1)
        return h0, h1
