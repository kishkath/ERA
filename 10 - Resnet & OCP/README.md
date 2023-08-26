**Session10: ResNet and One-cycle-Policy**: The session describes about the architecture of resnet and its variants followingly describing the most important way of tuning the hyper-parameter (learning-rate) in a intuitive way.

Ups n Downs are common: https://arxiv.org/abs/1803.09820
 
### Session 10 Assignment: 

🔏 Problem Statement: 
--------------------

<img src="https://github.com/kishkath/ERA/assets/60026221/81d819d6-4a49-424f-a788-c258b7021a4f" width = 720 height = 480>

💡 Define Problem:
------------------
 Develop the neural network similar to resnet architecture using identity connections. 
 
 With the help of One-cycle policy it has to achieve the 90% accuracy with in 24 Epochs.
 
🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── Resnet & OCP
    │   ├── models
    │   │   ├── S10_model.py: The Network Architecture designed to achieve 90% accuracy.
    │   ├── utility
    │   │   ├── dataset.py: Managing the data & retrieving it.
    │   │   ├── run.py:     Makes the model learn.
    │   │   ├── utils.py:   Contains the utilities required for the process.
    │   │   ├── visualize.py: Contains the code for visualizing.
    │   ├── OneCyclePolicy_Run.ipynb:  Execution of Network using one-cycle policy.
    └── README.md Details about the Process.


🔑 Model Architecture:
---------------------
 "C1 - (C2 + Residual) - C3 - (C4 + Residual) - MaxPool - FC - output"

  Total Parameter Count: 6,573,130

🔋 One Cycle Policy: 
-------------------

<p float="left">
  <img src="https://github.com/kishkath/ERA/assets/60026221/c08cfb91-7dd2-4ea4-915f-e3efcea8e292" width = 540 height = 360>
  <img src="https://github.com/kishkath/ERA/assets/60026221/2f8d7bb2-2284-45a5-8000-d652868b5668" width = 540 height = 360>
</p>


💊 Network Results: 
-------------------
 Trained the network for 24 Epochs with ADAM optimizer and CrossEntropyLoss fn.
 
 Achieved the desired accuracy at 24th Epoch.
 
      Epoch 24
      Train: Loss=0.1047 Batch_id=97 Accuracy=96.34: 100%|██████████| 98/98 [00:31<00:00,  3.10it/s]
      Test set: Average loss: 0.0006, Accuracy: 9098/10000 (90.98%)

 
 <img src="https://github.com/kishkath/ERA/assets/60026221/7e27f83b-c315-49ec-8e78-afb932199445" width = 720 height = 360>

 * Mis-classified Images:
 ------------------------

 <img src="https://github.com/kishkath/ERA/assets/60026221/d0fc17af-ced5-4c4c-8e3a-b95a11e8d36c" width = 720 height = 360>

