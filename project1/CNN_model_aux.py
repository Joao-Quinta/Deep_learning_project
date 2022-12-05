from torch import nn
from torch.nn import functional as F


class BinaryCNNAux(nn.Module):
    def __init__(self, dropout_rate):  ## defining the layers
        super().__init__()
        self.dropoutfc = nn.Dropout(p = dropout_rate)
        self.dropoutcnn = nn.Dropout(p = 0.0)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flatten0 = nn.Flatten(0)
        self.flatten1 = nn.Flatten(1)

        ### We need to convert to a sequential after we have a stable model
        # self.features = nn.Sequential()

        # Feature Extractors & Data Normalizers
        self.conv1 = nn.Conv2d(2, 64, kernel_size = 3, stride = 1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        # self.conv4 = nn.Conv2d(96, 128, kernel_size = 3, stride = 1, padding = 1)
        # self.batchnorm4 = nn.BatchNorm2d(128)

        # Classifiers & Output Layers
        self.fc1 = nn.Linear(256, 1)
        self.fc_aux = nn.Linear(256, 20)
        # self.fc1_bn = nn.BatchNorm1d(1)
        # self.fc2 = nn.Linear(100, 1)
        # self.fc2_bn = nn.BatchNorm1d()

    ## Generally, strides for convolution layers are 1 and for maxpools are 2
    def forward(self, x):
        ## Feature Extractors
        # print('First Input Shape: {}'.format(x.shape))
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = self.pool(x)
        x = self.dropoutcnn(x)
        # print('First Conv Layer Shape', x.shape)

        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool(x)
        x = self.dropoutcnn(x)
        # print('Second Conv Layer Shape', x.shape)

        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.max_pool2d(x, kernel_size=2, stride=2, dilation=(1, 1))
        x = self.dropoutcnn(x)
        # print('Third Conv Layer Shape', x.shape)

        ## Classifiers
        x = self.flatten1(x)
        # print('After Flattening', x.shape)
        x = self.dropoutfc(x)
        y = self.fc_aux(x)
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = self.batchnormfc1(x)
        # print('First Connected Layer: {} \n'.format(x.shape))
        x = self.flatten0(x)
        y = F.softmax(self.flatten1(y), dim = 1)
        # print('Final Output Shape {} \n'.format(x.shape))
        return x, y