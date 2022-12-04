from torch import nn
from torch.nn import functional as F
import torch
from CNN_model import BinaryCNN


# from CNN_weight_sharing_model import CNN_WeightSharing
class CNN_WeightSharing(nn.Module):
    def __init__(self):
        super().__init__()

        self.sharedConvNet = BinaryCNN(dropout_rate=0.2, initial_layers=1, final_layers=10)

        # Those layers are here for the final prediction, once the 2 CNN
        # have been merged
        self.linear1 = nn.Linear(20, 100)
        self.batchnorm1 = nn.BatchNorm1d(100)
        self.linear2 = nn.Linear(100, 20)
        self.batchnorm2 = nn.BatchNorm1d(20)
        self.linear3 = nn.Linear(20, 2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        print("x shape -> ", x.shape)
        # We split the input in 2 separate set of images
        image1 = x.narrow(1, 0, 1)
        print("image1 -> ", image1.shape)
        image2 = x.narrow(1, 1, 1)
        print("image2 -> ", image2.shape)

        # We call the CNN on thoses sets and merge the results
        output1 = self.sharedConvNet(image1)
        print("output1 -> ", output1.shape)
        output2 = self.sharedConvNet(image2)
        print("output2 -> ", output2.shape)
        outputCat = torch.cat((output1, output2), 1)
        print("outputCat -> ", outputCat.shape)

        # We apply our previously defined layers, as well as ReLu, batch
        # normalization and dropout
        outputCat = self.dropout(self.batchnorm1(F.relu(self.linear1(outputCat))))
        outputCat = self.dropout(self.batchnorm2(F.relu(self.linear2(outputCat))))
        outputCat = F.relu(self.linear3(outputCat))

        return outputCat
