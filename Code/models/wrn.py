import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0, use_norm=False, feature_norm=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)

        if use_norm:
            self.fc = NormedLinear(nChannels[3], num_classes)
        else:
            self.fc = nn.Linear(nChannels[3], num_classes)

        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.feature_norm = feature_norm


    def forward(self, x, return_feature=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if self.feature_norm:
            out = F.normalize(out, dim=1) * 25
        if return_feature:
            return self.fc(out), out
        return self.fc(out)


    # def feature_list(self, x):
    #     out_list = []
    #     out = self.conv1(x)
    #     out_list.append(out)
    #     out = self.block1(out)
    #     out_list.append(out)
    #     out = self.block2(out)
    #     out_list.append(out)
    #     out = self.block3(out)
    #     out_list.append(out)
    #     out = self.relu(self.bn1(out))
    #     out = F.avg_pool2d(out, 8)
    #     out = out.view(-1, self.nChannels)
    #     return self.fc(out), out_list

    def feature_list(self, x):
        out_list = []
        out = self.conv1(x)
        # out_list.append(out)
        out = self.block1(out)
        # out_list.append(out)
        out = self.block2(out)
        # out_list.append(out)
        out = self.block3(out)
        out_list.append(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out_list

    # function to extact a specific feature
    def intermediate_forward(self, x, layer_index):
        out = self.conv1(x)
        # if layer_index == 1:
        #     out = self.layer1(out)
        # elif layer_index == 2:
        #     out = self.block1(out)
        #     out = self.block2(out)
        # elif layer_index == 3:
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        return out





class classifier(nn.Module):
    '''
    Deep Neural Network with 3 hidden layers,
    each layer has batch normalization function locate before the activation function (ELU),
    the followed by the dropout function to reduce overfiting the reference and overcome batch effect
    '''
    def __init__(self, input_size, class_num,args):
        '''
        :param input_size: Input layer unit number (feature number)
        :param class_num: Number of different cell types.
        '''
        if input_size == None or class_num == None:
            raise ValueError("Must declare number of features and number of classes")
        super(classifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(0.5)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(0.5)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(0.1)
        self.layer4 = nn.Linear(32, class_num)



        # self.layer1 = nn.Linear(input_size, 128)
        # self.bn1 = nn.BatchNorm1d(128)
        # self.elu1 = nn.ELU()
        # self.dropout1 = nn.Dropout(0.5)
        # self.layer2 = nn.Linear(128, 64)
        # self.bn2 = nn.BatchNorm1d(64)
        # self.elu2 = nn.ELU()
        # self.dropout2 = nn.Dropout(0.5)
        # self.layer3 = nn.Linear(64, 32)
        # self.bn3 = nn.BatchNorm1d(32)
        # self.elu3 = nn.ELU()
        # self.dropout3 = nn.Dropout(0.1)
        # self.layer4 = nn.Linear(32, class_num)

        if args.loss=="logit_norm":
            self.normalize=True
            self.temp=args.temp
        else:
            self.normalize = False
        self.react=args.react
        print("REART {}:".format(self.react))

    def forward(self, x):
        '''
        :param x: forward calculation
        :return:
        '''
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.elu3(out)
        out = self.dropout3(out)
        # print(torch.max(out))
        if self.react and not self.training:
            out = out.clip(max=1)

        out = self.layer4(out)
        # print(out.size())

        if self.normalize:
            norms = torch.norm(out, p=2, dim=-1, keepdim=True) + 1e-16
            out = torch.div(out, norms)/self.temp



        return out

    def forward_feature(self, x):
        '''
        :param x: forward calculation
        :return:
        '''
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.elu3(out)
        out = self.dropout3(out)
        # print(torch.max(out))
        if self.react and not self.training:
            out = out.clip(max=1)

        # out = self.layer4(out)
        #
        # if self.normalize:
        #     norms = torch.norm(out, p=2, dim=-1, keepdim=True) + 1e-16
        #     out = torch.div(out, norms)/self.temp



        return out

    def forward_threshold(self,x,threshold=1e10):
        out = self.layer1(x)
        out = self.bn1(out)
        out = self.elu1(out)
        out = self.dropout1(out)
        out = self.layer2(out)
        out = self.bn2(out)
        out = self.elu2(out)
        out = self.dropout2(out)
        out = self.layer3(out)
        out = self.bn3(out)
        out = self.elu3(out)
        # out = self.dropout3(out)
        out=out.clip(max=threshold)
        out = self.layer4(out)
    def forward_feature_list(self,x,threshold=1e10):
        list=[]
        out = self.layer1(x)

        out = self.bn1(out)
        out = self.elu1(out)

        list.append(F.normalize(out, p=2, dim=1))

        out = self.dropout1(out)


        out = self.layer2(out)

        out = self.bn2(out)
        out = self.elu2(out)
        list.append(F.normalize(out, p=2, dim=1))
        out = self.dropout2(out)
        out = self.layer3(out)


        out = self.bn3(out)
        out = self.elu3(out)
        list.append(F.normalize(out, p=2, dim=1))
        # out = self.dropout3(out)
        out=out.clip(max=threshold)
        out = self.layer4(out)
        # list.append(F.normalize(out,p=2,dim=1))
        list=torch.cat(list,dim=1)
        return list
    def set_react(self,react):
        self.react=react