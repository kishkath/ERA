**Session9: Advanced Convolutions & Augmentations**: The session describes about the various types of convolutions specifying their advantages & disadvantages. It also introduces a technique to reduce over-fitting issue known as Augmentating data using PyTorch library 
Albumentations. 

### Session 9 Assignment: 

A. Problem Statement: 
Write a new network thathas the architecture to C1C2C3C40 (No MaxPooling, but 3 convolutions, where the last one has a stride of 2 instead)

(If you can figure out how to use Dilated kernels here instead of MP or strided convolution, then 200pts extra!)

total RF must be more than 44
one of the layers must use Depthwise Separable Convolution
one of the layers must use Dilated Convolution
use GAP (compulsory):- add FC after GAP to target #of classes (optional)
use albumentation library and apply:
horizontal flip
shiftScaleRotate
coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
achieve 85% accuracy, as many epochs as you want. Total Params to be less than 200k.
