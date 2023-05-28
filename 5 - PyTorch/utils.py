import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

class Load_dataset:
  def __init__(self,BATCH_SIZE,shuffle=True):
    self.BATCH_SIZE = BATCH_SIZE
    self.shuffle=True
 

  def get_dataset(self):
    train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.CenterCrop(22),],p=0.1),
      transforms.Resize((28,28)),
      transforms.RandomRotation((-15.,15.),fill=0),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,)),
    ])

    test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,))
    ])

    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    kwargs = {'batch_size': self.BATCH_SIZE, 'shuffle': self.shuffle, 'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader,test_loader

class Plots:
  def __init__(self,num_images=None,loaded_data=None,metrics=None):
    self.num_images = num_images
    self.loaded_data = loaded_data
    self.metrics = metrics

  def plot_images(self):
    batch_data, batch_label = next(iter(self.loaded_data))
    fig = plt.figure()
    if self.num_images % 2!=0:
      self.num_images -= 1
    self.num_rows = self.num_images//3

    fig = plt.figure(figsize=(15,7))
    counter = 0
    for i in range(self.num_images):
      sub = fig.add_subplot(self.num_rows,3, i + 1)
      im2display = (np.squeeze(batch_data[i].permute(2, 1, 0)))
      sub.imshow(im2display.cpu().numpy())
      sub.set_title(batch_label[i])
      sub.axis('off')

    plt.tight_layout()
    plt.axis('off')
    plt.show()
    

  def plot_metrics(self):
    if self.metrics == None:
      print("Please provide the metric values, unable to view them!")
    else:
      train_losses,train_acc,test_losses,test_acc = self.metrics
      fig,axs = plt.subplots(2,2,figsize=(15,10))
      axs[0, 0].plot(train_losses)
      axs[0, 0].set_title("Training Loss")
      axs[1, 0].plot(train_acc)
      axs[1, 0].set_title("Training Accuracy")
      axs[0, 1].plot(test_losses)
      axs[0, 1].set_title("Test Loss")
      axs[1, 1].plot(test_acc)
      axs[1, 1].set_title("Test Accuracy")
