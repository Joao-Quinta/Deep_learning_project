from torch import nn
from torch.nn import functional as F


## Binary Classification Model (2 input -> 1 output)
## This is the initial CNN model that we decided on without weight sharing or aux loss
## Dropout layers on CNN were removed as it was found it lowered performance
## One dropout layer before the final FC layer with a rate to be passed as a parameter

class BinaryCNNLegacy(nn.Module):
    def __init__(self, dropout_rate):  ## defining the layers
        super().__init__()
        self.dropoutfc = nn.Dropout(p=dropout_rate)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten0 = nn.Flatten(0)
        self.flatten1 = nn.Flatten(1)

        # Feature Extractors & Data Normalizers 
        self.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)

        # Classifiers & Output Layers
        self.fc1 = nn.Linear(256, 1)

    ## Generally, strides for convolution layers are 1 and for maxpools are 2
    def forward(self, x):
        ## Feature Extractors        
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        # print('First Conv Layer Shape', x.shape)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        # print('Second Conv Layer Shape', x.shape)
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), kernel_size=2, stride=2, dilation=(1, 1))
        # print('Third Conv Layer Shape', x.shape)

        ## Classifiers
        x = self.flatten1(x)
        # print('After Flattening', x.shape)
        x = self.dropoutfc(x)
        x = self.fc1(x)
        # print('First Connected Layer: {} \n'.format(x.shape))
        x = self.flatten0(x)
        # print('Final Output Shape {} \n'.format(x.shape))
        # print('Final Output', x)
        return x
