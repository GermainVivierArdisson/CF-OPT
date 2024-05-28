import torch
import torch.nn as nn
from torchvision.models import resnet18

class PartialResNet(nn.Module):
    """
    ResNet18 model with Max Pooling for the shortest paths on Warcraft maps pipeline.
    """

    def __init__(self, k):
        super(PartialResNet, self).__init__()
        # init resnet 18
        resnet = resnet18(pretrained=False)
        # first five layers of ResNet18
        self.conv1 = resnet.conv1
        self.bn = resnet.bn1
        self.relu = resnet.relu
        self.maxpool1 = resnet.maxpool
        self.block = resnet.layer1
        # conv to 1 channel
        self.conv2 = nn.Conv2d(64, 1, kernel_size=(1, 1),
                               stride=(1, 1), padding=(1, 1), bias=False)
        # max pooling
        self.maxpool2 = nn.AdaptiveMaxPool2d((k, k))

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.maxpool1(h)
        h = self.block(h)
        h = self.conv2(h)
        out = self.maxpool2(h)
        # reshape for optmodel
        out = torch.squeeze(out, 1)
        out = out.reshape(out.shape[0], -1)
        return out

class LinearRegression(nn.Module):
    """
    Linear prediction model for the shortest paths on a grid pipeline.
    Attributes:
        num_feat (int): dimension of the contextual features, defines size of the input.
        grid (tuple of int): grid on which shortest paths are computed, defines size of the ouput.
    """
    
    def __init__(self, num_feat, grid):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(num_feat, (grid[0]-1)*grid[1]+(grid[1]-1)*grid[0])
    
    def forward(self, x):
        return self.linear(x)
    
class LinearRegression_KnapSack(nn.Module):
    """
    Linear prediction model for the shortest paths on a grid pipeline.
    Attributes:
        p (int): dimension of the contextual features, defines size of the input.
        m (int): number of items, defines size of the ouput.
    """
    
    def __init__(self, p, m):
        super(LinearRegression_KnapSack, self).__init__()
        self.linear = nn.Linear(p, m)
    
    def forward(self, x):
        return self.linear(x)
    
class FeedForwardNN(nn.Module):
    """
    Dense feed-forward neural network for the shortest paths on a grid pipeline.
    Attributes:
        in_dim (int): dimension of the contextual features.
        out_dim (int): dimension of the predicted costs.
        n_layers (int): number of layers of the neural network.
    """
    
    def __init__(self, in_dim, out_dim, n_layers):
        super(FeedForwardNN, self).__init__()
        layers = []
        if n_layers > 1:
            layers.append(nn.Linear(in_dim, in_dim))
            layers.append(nn.ReLU()) 
        else:
            layers.append(nn.Linear(in_dim, out_dim))

        for i in range(1, n_layers):
            if i < n_layers-1:
                layers.append(nn.Linear(in_dim, in_dim))
                layers.append(nn.ReLU())  
            else:
                layers.append(nn.Linear(in_dim, out_dim))
            

        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)