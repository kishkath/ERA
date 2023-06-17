import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


## Model
## Model

class NetArch1(nn.Module):
    def __init__(self):
        super(NetArch1, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.ReLU())
        # out_features = (28+(2*1)-3)/1 + 1 = 28
        # Jin = 1, S = 1, Jout = 1 ,
        # Rf_out = Rin + (K-1)*Jin = 1 + (3-1)*1 = 3
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (28-3)/1 + 1 = 26
        Jin = 1, S = 1, Jout = 1 
        Rf_out = Rin + (K-1)*Jin = 3 + (3-1)*1 = 5
        """
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (26-3) + 1 = 24 
        Jin = 1, S = 1, Jout = 1 
        Rf_out = Rin + (K-1)*Jin = 5 + (3-1)*1 = 7

        * As Input Size is 28, We can think of level1 edges & gradients would have been determined
           with MaxPool we can scale down the computation by reducing the image-size,
        """
        self.maxPool = nn.MaxPool2d((2, 2))
        """
        out_features = (24-2)/2 + 1 = 12 
        Jin = 1, S = 2, Jout = 2
        Rf_out = Rin + (K-1)*Jin = 7 + (2-1)*1 = 8 
        """
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (12-3)/1 + 1 = 10 
        Jin = 2, S = 1, Jout = 2 
        Rf_out = Rin + (K-1)*Jin = 8 + (3-1)*2 = 12 

        * Lets use 1x1 for ease of parameter computation
        """
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 128, 1, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (10-1) + 1 = 10
        Jin = 2, S = 1, Jout = 2
        Rf_out = Rin + (K-1)*Jin = 12 + (1-1)*2 = 12
        """
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (10-3)/1 + 1 = 8
        Jin = 2, S = 1, Jout = 2
        Rf_out = Rin + (K-1)*Jin = 12 + (3-1)*2 = 16
        """
        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (8-3) + 1 = 6 
        Jin = 2, S = 1, Jout = 2 
        Rf_out = Rin + (K-1)*Jin = 16 + (3-1)*2 = 20

        * Lets again use 1x1 for dealing with fewer number of parameters.
        """
        self.conv8 = nn.Sequential(
            nn.Conv2d(512, 256, 1, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (6-1)+1 = 6
        Jin = 2, S = 1, Jout = 2 
        Rf_out = Rin + (K-1)*Jin = 20 + (1-1)*2 = 20
        """
        self.conv9 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=0, bias=False),
            nn.ReLU())
        """
        out_features = (6-3)/1 + 1 = 4
        Jin = 2, S = 1, Jout = 2
        Rf_out = Rin + (K-1)*Jin = 20 + (3-1)*2 = 24
        """
        self.linear1 = nn.Linear(4 * 4 * 128, 512)
        self.linear2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxPool(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = x.view(-1, 4 * 4 * 128)
        x = self.linear1(x)
        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)


def return_summary(device, INPUT_SIZE):
    model = NetArch1().to(device)
    return summary(model, input_size=INPUT_SIZE)
