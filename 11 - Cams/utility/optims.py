import json
import torch.optim as optim 
from utility.utils import load_config 

config_data = load_config()
optimizer_variant = config_data['optimizer'] 
LR = config_data['learning_rate']
momentum = config_data['momentum']
decay = config_data['weight_decay']
scheduler_variant = config_data['scheduler']
ocp = config_data['ocp']


def hyper_parameters(model,ocp=False):
    if optimizer_variant == "SGD":
        optimizer = optim.SGD(model.parameters(),lr=LR,momentum=momentum,weight_decay=0.00)
    elif optimizer_variant == "Adam":
        optimizer = optim.Adam(model.parameters(),lr=LR,momentum=momentum,weight_decay=decay)

    if ocp==False:
        if scheduler_variant == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        elif scheduler_variant == 'steplr':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1, verbose=True)

    if ocp==True:
        return optimizer 
    else:
        return optimizer,scheduler


