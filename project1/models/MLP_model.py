from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, dropout_rate):  ## defining the layers
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.flatten_dim_0 = nn.Flatten(0)
        self.flatten_dim_1 = nn.Flatten(1)

        # Fully Connected Layers
        self.fc1 = nn.Linear(2 * 14 * 14, 1024)  ## (2x14x14) images
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)  ## output layers
        self.fc_aux = nn.Linear(256, 20)

    ## Generally, strides for convolution layers are 1 and for maxpools are 2
    ## Uncomment the prints for debuggingd
    def forward(self, x):
        x = self.flatten_dim_1(x)
        x = self.dropout(x)
        # print('Flattened Input Shape: ', x.shape)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        # print('First FC Layer Shape', x.shape)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        # print('Second FC Layer Shape', x.shape)
        x = F.relu(self.dropout(self.fc3_bn(self.fc3(x))))
        # print('Third FC Layer Shape', x.shape)
        y = self.fc_aux(x)
        x = self.fc4(x)
        
        x = self.flatten_dim_0(x)
        y = self.flatten_dim_1(y)
        # print('Final Output Shape {} \n'.format(x.shape))
        return x, y
