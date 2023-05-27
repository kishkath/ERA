import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class Load_dataset:
  def __init__(self,BATCH_SIZE,shuffle=True):
    self.BATCH_SIZE = BATCH_SIZE
    self.shuffle=True
  def get_train_transforms(self):
    train_transforms = transforms.Compose([
      transforms.RandomApply([transforms.CenterCrop(22),],p=0.1),
      transforms.Resize((28,28)),
      transforms.RandomRotation((-15.,15.),fill=0),
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,)),
    ])
    return train_transforms

  def get_test_transforms(self):
    test_transforms = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,),(0.3081,))
    ])
    return test_transforms

  def get_dataset(self,train_transforms=get_train_transforms(), test_transforms=get_test_transforms()):
    train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
    test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)

    kwargs = {'batch_size': self.BATCH_SIZE, 'shuffle': self.shuffle, 'num_workers': 2, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)

    return train_loader,test_loader)

class Plots:
  def __init__(self,num_images,loaded_data,metrics=None):
    self.num_images = num_images
    self.loaded_data = loaded_data
    self.metrics = metrics

  def plot_images(self):
    batch_data, batch_label = next(iter(self.loaded_data))
    fig = plt.figure(figsize=(12,30))
    if self.num_images % 2!=0:
      self.num_images -= 1

    self.num_rows = self.num_images//6

    for i in range(self.num_images):
      plt.subplot(self.num_rows, 6, i + 1)
      plt.tight_layout()
      plt.imshow(batch_data[i].squeeze(0), cmap='gray')
      plt.title(batch_label[i].item())
      plt.xticks([])
      plt.yticks([])
    return ""

  def plot_metrics(self):
    if self.metrics == None:
      print("Please provide the metric values, unable to view them!")
        return ""
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
