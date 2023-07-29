import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from time import time 
import time
from torch.optim.lr_scheduler import OneCycleLR
import argparse 

from utility.cams import plotting_gradCams

import os
from models.resnet import *
from utility.utils import load_config,allot_device 
from utility.visualizers import plot_metrics, Plots
from utility.modelling import Performance,scores
from utility.dataset import loader
from utility.optims import hyper_parameters



parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--max_lr', type=float, help='learning rate')
parser.add_argument('--restart', '-restart', action='store_true',
                    help='restart from first')
parser.add_argument("--resume","-resume",action="store_true",help="resume from checkpoint")
args = parser.parse_args()

config_data = load_config()
VISUALIZE = config_data['visualize']
BATCH_SIZE = config_data['batch_size']
NUM_EPOCHS = config_data['epochs']
NUM_IMAGES = config_data['num_plot_dataset_images']
random_seed_value = config_data['random_seed_value']
ocp = config_data['ocp']

device = allot_device(random_seed_value)
train_loader,test_loader = loader.load_data(BATCH_SIZE)
if VISUALIZE == True:
    Plots(None,NUM_IMAGES,train_loader).plot_images()
    Plots(None,NUM_IMAGES,test_loader).plot_images()
        

model = ResNet18().to(device) 
criterion = nn.CrossEntropyLoss()
optimizer,scheduler = hyper_parameters(model)
if ocp==True:
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=5/NUM_EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )

perf = Performance(device,model,(train_loader,test_loader),optimizer,criterion,[False,0.01],scheduler=scheduler)
def do_train(model,criterion,optimizer,scheduler):
    try:
        for epoch in range(1, NUM_EPOCHS+1):
            print(f'Epoch {epoch}')
            perf.train()
            perf.test()

        if VISUALIZE == True:
            metrics = scores()
            plot_metrics(metrics)
            Plots((model,test_loader,device,NUM_IMAGES)).mis_classified()
    except Exception as e:
        print("Error",e)


if args.restart:
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

begin_time = time.time()
do_train(model,criterion,optimizer,scheduler)
end_time = time.time()
processed_time = round((end_time - begin_time)/60,2)
print(f"Training took {processed_time} mins.")
plotting_gradCams(NUM_IMAGES)