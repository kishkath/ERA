**Session12: PyTorch Lightning**: The session describes about the CAM Visualizations using GRADCAM followingly describes the nature of hyperparameters and its usage.

Lighten up: https://lightning.ai/
 
### Session 12 Assignment: 

🔏 Problem Statement:

--------------------

         Move your S10 assignment to Lightning first and then to Spaces such that:
         
         (You have retrained your model on Lightning)
         
         You are using Gradio
         
         Your spaces app has these features:
         
         ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
         ask whether he/she wants to view misclassified images, and how many
         
         allow users to upload new images, as well as provide 10 example images
         
         ask how many top classes are to be shown (make sure the user cannot enter more than 10)
         
          

💡 Define Problem:
------------------
 Convert pytorch to pytorch lightning. 
 
🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── Space App_Development
    │   ├── lightning_run.ipynb: Contains the pytorch-lightning code for the CIFAR10 dataset using custom resnet model.
    └── README.md Details about the Process.

  Process:
  -------
  * The process tends to convert pytorch to pytorch lightning


🔑 Model Architecture:
---------------------
 "Custom Resnet": Session10 - Resnet & OCP


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
         
         ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
         ┃        Test metric        ┃       DataLoader 0        ┃
         ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
         │          val_acc          │    0.8730000257492065     │
         │         val_loss          │    0.42876118421554565    │
         └───────────────────────────┴───────────────────────────┘
         [{'val_loss': 0.42876118421554565, 'val_acc': 0.8730000257492065}]

