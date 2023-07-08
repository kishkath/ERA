import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary



class NetArch(nn.Module):
    def __init__(self):
        super(NetArch, self).__init__()

        ## Prep Layer
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU())

        ## Layer1
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.residue1 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU())

        ## Layer2
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU())

        ## Layer3
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.residue2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU())

        ## Layer4
        self.maxPool1 = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        y = self.residue1(x)

        x = x + y
        x = self.conv2(x)
        x = self.conv3(x)
        y = self.residue2(x)

        x = x + y

        x = self.maxPool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


        return F.log_softmax(x, dim=-1)



def return_summary(model,device, INPUT_SIZE):
    return summary(model, input_size=INPUT_SIZE)