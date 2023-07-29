**Session11: Cams and hyperparameter tuning**: The session describes about the CAM Visualizations using GRADCAM followingly describes the nature of hyperparameters and its usage.

Model too observes: https://jacobgil.github.io/pytorch-gradcam-book/introduction.html
 
### Session 11 Assignment: 

🔏 Problem Statement:

--------------------

<img src="https://github.com/kishkath/ERA/assets/60026221/908b159c-9b0d-4b79-91c6-847ee47fc133" width = 720 height = 480>

💡 Define Problem:
------------------
 Develop the neural network similar with CAM visualizations. 
 .
 
🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── Advanced Convolutions & Augmentations
    │   ├── models
    │   │   ├── resnet.py: The Network Architecture designed to achieve 90% accuracy.
    │   ├── utility
    │   │   ├── cams.py  : Contains the code for GRADCAM visuals
    │   │   ├── dataset.py: Managing the data & retrieving it.
    │   │   ├── modelling.py: Contains the code for making the model learn.
    │   │   ├── optims.py:  Upon the chosen optimizer, the file returns respective optimizer & scheduler
    │   │   ├── utils.py:   Contains the utilities codes such as device allocation and loading json file with inputs required for the process.
    │   │   ├── visualizers.py: Contains the code for visualizing.
    │   ├── OneCyclePolicy_Run2.ipynb:  Execution of Network using one-cycle policy.
    │   ├── lr_finder.py: Contains the code for finding the learning rate using one-cycle policy.
    │   ├── run.py:  Contains the code for training the model using obtained lr from lr_finder.py
    │   ├── requirements.txt: Contains the necessary libraries
    └── README.md Details about the Process.

  Process:
  -------
  * The process begins with the provision of relative inputs to hyperparameters in the 'Configs.json' (present in utility folder).

  * Once the inputs are set up, relax.

  * Clone the repo 

  * Install requirements.txt

  * Run lr_finder.py : Here, we will get a suggested LR suitable for model. Provide it in next step. 
              
               !python run lr_finder.py

  * Run run.py with the max_lr as lr_finder.py's suggested LR. 
              
              !python run --max_lr==obtained_suggested_LR. (If only OCP is being used else no need of max_lr argument)

      * If want to restart training the model use command: 
         
             !python run --restart 

      * If want to resume training the model use command: 
             
             !python run --resume 

  * All Images will be stored in images directory & model in checkpoint directory.

🔑 Model Architecture:
---------------------
 "ResNet18": https://github.com/kuangliu/pytorch-cifar


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


 <img src="https://github.com/kishkath/ERA/assets/60026221/d93b3435-29d5-4967-b9ce-172f3d5975fe" width = 720 height = 360>

 * Mis-classified Images:
 ------------------------

 <img src="https://github.com/kishkath/ERA/assets/60026221/3478d7d2-d7e9-4b4b-ad3b-d22bf808e7fb" width = 720 height = 360>

