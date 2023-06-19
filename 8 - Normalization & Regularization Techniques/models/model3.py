import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


## Model
## Model
# C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
class NetArch(nn.Module):
    def __init__(self):
        super(NetArch, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.GroupNorm(2,16),
            nn.ReLU())

        """
        out_features = (32+2-3)/1 +  1 = 32
        Jin = 1, S = 1, Jout = 1 
        Rfout = Rin + (K-1)*Jin = 1  + (3-1)*1 = 3
        """
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.GroupNorm(2,16),
            nn.ReLU())
        """
        out_features = (32+2-3)/1 + 1 = 32
        Jin = 1, S = 1, Jout = 1,
        Rfout = Rin + (K-1)*Jin = 3 + (3-1)*1 = 5
        """

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(16, 20, 1, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = 28 
        Rfout = 5
        """

        self.maxPool1 = nn.MaxPool2d((2, 2))
        """
        out_features = (30-2)/2 + 1 = 15 
        Jin = 1, S = 2, Jout = 2 
        Rfout = Rin + (K-1)*Jin = 5 + (2-1)*1 = 6 
        """

        self.conv_block4 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1, bias=False),
            nn.GroupNorm(2,20),
            nn.ReLU())
        """
        out_features = (15+2-3)+1 = 15 
        Jin = 2, S = 1, Jout = 2
        Rfout = Rin + (K-1)*Jin = 6 + (3-1)*2 = 10
        """
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(20, 20, 3, padding=1, bias=False),
            nn.GroupNorm(2,20),
            nn.ReLU())
        """
        out_features = (15+2-3) + 1 = 15 
        Jin = 2, S = 1, Jout = 2
        Rfout = Rin + (K-1)*Jin = 10 + (3-1)*2 = 14
        """
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(20, 32, 3, padding=0, bias=False),
            nn.GroupNorm(2,32),
            nn.ReLU())
        """
        out_features = (15-3) + 1 = 13
        Jin = 2, S = 1, Jout = 2 
        Rfout = 18
        """
        self.conv_block7 = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = 13 ,Rfout = 18 

        """
        self.maxPool2 = nn.MaxPool2d((2, 2))
        """
        out_features =  (13-2)/2 + 1 = 6 
        Jin = 2, S = 2, Jout = 4 
        Rfout = 18 + (2-1)*2 = 22
        """
        self.conv_block8 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.GroupNorm(2,32),
            nn.ReLU())
        """
        out_features =  (6-3)  + 1 = 6 
        Jin = 4, S= 1, Jout = 4
        Rfout = 22 + (3-1)*4 = 30
        """
        self.conv_block9 = nn.Sequential(
            nn.Conv2d(32, 48, 3, padding=1, bias=False),
            nn.GroupNorm(2,48),
            nn.ReLU())
        """
        out_fetures = 6
        Jin = 4, S = 1, Jout = 4 
        Rfout = 30 + 8 = 38 
        """
        self.conv_block10 = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=0, bias=False),
            nn.GroupNorm(2,16),
            nn.ReLU())
        """
        out_features = (6-3) + 1 = 4
        Jin = 4, S = 1, Jout = 4
        Rfout = 38 + 8 = 46
        """
        self.gap = nn.AvgPool2d(4)
        """
        out_features = (4-4) + 1 = 1 
        Jin = 4, S = 1, Jout = 4 
        Rfout = 46 + (4-1)*4 = 58
        """
        self.conv_block11 = nn.Sequential(
            nn.Conv2d(16, 10, 1, padding=0, bias=False))

    def forward(self, x):
        x = self.conv_block1(x)
        x = x + self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.maxPool1(x)
        x = x + self.conv_block4(x)
        x = x + self.conv_block5(x)
        x = self.conv_block6(x)
        x = self.conv_block7(x)
        x = self.maxPool2(x)
        x = self.conv_block8(x)
        x = self.conv_block9(x)
        x = self.conv_block10(x)
        x = self.gap(x)
        x = self.conv_block11(x)
        x = x.view(-1, 10)

        return F.log_softmax(x, dim=-1)


def return_summary(model,device, INPUT_SIZE):
    return summary(model, input_size=INPUT_SIZE)
