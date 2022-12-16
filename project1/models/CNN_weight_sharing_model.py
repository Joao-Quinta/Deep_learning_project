from torch import nn
from torch.nn import functional as F
import torch

# from CNN_weight_sharing_model import CNN_WeightSharing
class BinaryCNN(nn.Module):
    def __init__(self, dropout_rate): ## defining the layers
        super().__init__()
        self.dropoutfc = nn.Dropout(p = dropout_rate)
        self.dropoutcnn = nn.Dropout(0.0)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.flatten0 = nn.Flatten(0)
        self.flatten1 = nn.Flatten(1)
        
        # Feature Extractors & Data Normalizers 
        self.conv1 = nn.Conv2d(1, 64, kernel_size = 3, stride = 1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv3_bn = nn.BatchNorm2d(256)
        
        # Classifiers & Output Layers
        self.fc1 = nn.Linear(256, 10)
        
    ## Generally, strides for convolution layers are 1 and for maxpools are 2
    def forward(self, x): 
        ## Feature Extractors        
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.dropoutcnn(x)    
        # print('First Conv Layer Shape', x.shape)
        x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.dropoutcnn(x)    
        # print('Second Conv Layer Shape', x.shape)
        x = F.max_pool2d(F.relu(self.conv3_bn(self.conv3(x))), kernel_size= 2, stride = 2, dilation = (1, 1))
        x = self.dropoutcnn(x)
        # print('Third Conv Layer Shape', x.shape)
        
        ## Classifiers
        x = self.flatten1(x)
        # print('After Flattening', x.shape)
        x = self.dropoutfc(x)
        x = self.fc1(x)
        # x = F.relu(self.fc1(x))
        # x = self.batchnormfc1(x)
        # print('First Connected Layer: {} \n'.format(x.shape))
        x = self.flatten0(x)
        # print(y)
        # print('after softmax', y)
        # print('Final Output Shape {} \n'.format(x.shape))
        # print('Output', x)
        return x
    
    
class BinaryCNNSharing(nn.Module):
    def __init__(self, dropout_rate): ## defining the layers
        super().__init__()
        self.dropoutfc = nn.Dropout(p = dropout_rate)
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
        
        ## Concatenation of two outputs
        x = torch.cat((out1, out2), 1)
        # print('concatenated', x)
        ## Classifiers
        x = self.dropoutfc(x)
        y = x  
        x = self.fc1(x)
        x = self.flatten0(x)
        # x = self.flatten1(x)
        # # print('After Flattening', x.shape)
        # x = self.dropoutfc(x)
        # x = self.fc1(x)
        # # x = F.relu(self.fc1(x))
        # # x = self.batchnormfc1(x)
        # # print('First Connected Layer: {} \n'.format(x.shape))
        # x = self.flatten0(x)
        # # print(y)
        # # print('after softmax', y)
        # print('Final Output Shape {} \n'.format(x.shape))
        # print('Output', x)
        return x, y