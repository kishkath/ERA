from utility.utils import allot_device 
from utility.optims import hyper_parameters
from utility.dataset import loader 
from models.resnet import *
from torch_lr_finder import LRFinder 
from copy import deepcopy
from utility.utils import load_config 
import matplotlib.pyplot as plt

config_data = load_config()
BATCH_SIZE = config_data['batch_size']
device = allot_device(42)
train_loader,test_loader = loader.load_data(BATCH_SIZE)
lr_model = ResNet18().to(device)

optimizer = hyper_parameters(lr_model,True)
criterion = nn.CrossEntropyLoss()
lr_finder = LRFinder(lr_model,optimizer,criterion,device="cuda")
print(lr_finder.range_test(train_loader,end_lr=10,num_iter=200,step_mode="exp"))
lr_finder.plot()
plt.plot("plot_lr.jpg")
lr_finder.reset()