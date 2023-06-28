import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary


## Model
## Model
# C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10
import torch.nn as nn
class NetArch(nn.Module):
    def __init__(self):
        super(NetArch,self).__init__()
        
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,32,3,stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
         
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
           
            nn.Conv2d(32,128,3,padding=1,dilation=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64,64,3,padding=1,groups=128),
            nn.Conv2d(64,64,1),
        
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.Conv2d(64,64,3,stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(64,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32,16,3),
            
            nn.AvgPool2d(2))

        self.fc = nn.Linear(1*1*16,10)
        self.dropout = nn.Dropout2d(0.05)
    def forward(self,x):
        x = self.conv_block1(x)
        x = self.dropout(x)
        x = self.conv_block2(x)
        x = self.dropout(x)
        x = self.conv_block3(x)
        x = self.dropout(x)
        x = self.conv_block4(x)
        x = x.reshape(-1,32)
        x = self.fc(x)

        return F.log_softmax(x,dim=-1)
            


def return_summary(model,device, INPUT_SIZE):
    return summary(model, input_size=INPUT_SIZE)
