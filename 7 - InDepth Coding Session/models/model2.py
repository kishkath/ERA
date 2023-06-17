import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


## Model
## Model

class NetArch2(nn.Module):
    def __init__(self):
        super(NetArch2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.ReLU())

        """
        out_features = (28+2-3)/1 + 1 = 28
        Jin = 1, S = 1, Jout = 1 
        Rf_out = Rin + (K-1)*Jin = 1 + (3-1)*1 = 3
        """
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 20, 3, bias=False),
            nn.ReLU())
        """
        out_features = (28-3)/1 + 1 = 26
        Jin = 1, S = 1, Jout = 1
        Rf_out = Rin + (K-1)*Jin = 3 + (3-1)*1 = 5
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(20, 32, 3, bias=False),
            nn.ReLU())
        """
        out_features = (26-3)/1 + 1 = 24
        Jin = 1, S = 1, Jout = 1
        Rf_out = Rin + (K-1)*Jin = 5 + (3-1)*1 = 7
        """
        self.maxPool1 = nn.MaxPool2d((2, 2))
        """
        out_features = (24-2)/2 + 1 = 12 
        Jin = 1, S=2, Jout = 2 
        Rf_out = Rin + (K-1)*Jin = 7 + (2-1)*1 = 8 
        """
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, bias=False),
            nn.ReLU())
        """
        out_features = (12-3) + 1 = 10
        Jin = 2, S = 1, Jout = 2 
        Rf_out = Rin + (K-1)*Jin = 8 + (3-1)*2 = 12 
        """
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.ReLU())
        """
        out_features = (10-1) + 1 = 10 
        Jin = 2, S = 1, Jout = 2
        Rf_out = Rin + (K-1)*Jin = 12 + (1-1)*2 = 12
        """

        self.conv6 = nn.Sequential(
            nn.Conv2d(32, 32, 3, bias=False),
            nn.ReLU())
        """
        out_features = (10-3) + 1 = 8
        Jin = 2, S = 1, Jout =2
        Rf_out = Rin + (K-1)*Jin = 12 + (3-1)*2 = 16
        """

        self.linear1 = nn.Linear(8 * 8 * 32, 512, bias=False)
        self.linear2 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxPool1(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 8 * 8 * 32)
        x = self.linear1(x)
        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)

def return_summary(device, INPUT_SIZE):
    model = NetArch2().to(device)
    return summary(model, input_size=INPUT_SIZE)
