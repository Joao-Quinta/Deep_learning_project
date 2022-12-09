import torch
from torch import nn
from torch.nn import functional as F

from models.CNN_1_layer_model import BinaryCNN


class BinaryCNNSharing(nn.Module):
    def __init__(self, dropout_rate):  ## defining the layers
        super().__init__()
        self.dropoutfc = nn.Dropout(p=dropout_rate)
        self.flatten0 = nn.Flatten(0)
        self.flatten1 = nn.Flatten(1)

        # Feature Extractors & Data Normalizers
        self.sharedConvNet = BinaryCNN(dropout_rate=0.0)

        # Classifiers & Output Layers
        self.fc1 = nn.Linear(20, 1)
        # self.fc1_bn = nn.BatchNorm1d(1)
        # self.fc2 = nn.Linear(100, 1)
        # self.fc2_bn = nn.BatchNorm1d()

    ## Generally, strides for convolution layers are 1 and for maxpools are 2
    def forward(self, x):
        ## Feature Extractors
        # We split the input in 2 separate set of images
        img1 = x.narrow(1, 0, 1)
        img2 = x.narrow(1, 1, 1)

        out1 = self.sharedConvNet(img1)
        out1 = out1.view(out1.size(-1) // 10, 10)
        # print('output 1 shape', out1.shape, out1)

        out2 = self.sharedConvNet(img2)
        out2 = out2.view(out2.size(-1) // 10, 10)
        # print('output 2 shape', out2.shape, out2)

        x = torch.cat((out1, out2), 1)
        # print('kh kh', x.shape)
        x = self.dropoutfc(x)
        x = self.fc1(x)
        # x = torch.sigmoid(x)
        # x = torch.max(x, 1)[0]
        x = self.flatten0(x)
        return x
