**Session13: YOLO**: The session describes about the YOLO-v3 ands its family.

You Look Only Once(YOLOv3): https://github.com/eriklindernoren/PyTorch-YOLOv3 
 
### Session 13 Assignment: 

🔏 Problem Statement:

--------------------

          Move the code to PytorchLightning
          
          Train the model to reach such that all of these are true:
          
          Class accuracy is more than 75%
          
          No Obj accuracy of more than 95%
          
          Object Accuracy of more than 70% (assuming you had to reduce the kernel numbers, else 80/98/78)
          
          Ideally trailed till 40 epochs
          
          Add these training features:
          
          Add multi-resolution training - the code shared trains only on one resolution 416
          
          Add Implement Mosaic Augmentation only 75% of the times
          
          Train on float16
          
          GradCam must be implemented.
          
          Things that are allowed due to HW constraints:
          
          Change of batch size
          
          Change of resolution
          
          Change of OCP parameters
         
          

💡 Define Problem:
------------------
 Run the Yolo with Pytorch-lightning.
 
🚦 Follow-up Process:
-----------------
 The directory structure describes in following way:

    Directory: 
    ---------
    ├── YOLO
    │   ├── config.py: Code about the configurations.
    │   ├── dataset.py: Contains how to retrieve the dataset and split to train, test,val
    │   ├── loss.py: Contains the loss function and its parameters.
    │   ├── model.py: Contains the CNN Network with 3-4 blocks.
    │   ├── train.py: Contains the training code.
    │   ├── utils.py: Contains all the utility codes.
    │   ├── yolov3_lightning_run.ipynb: Yolov3 run with Pytorch-lightning Framework
    └── README.md Details about the Process.

  Process:
  -------
  * The process tends to convert pytorch to pytorch lightning
  * Solution:
  * Built the hugging face spaces app, using Gradio. link:  [https://huggingface.co/spaces/kishkath/cifar10C](https://huggingface.co/spaces/kishkath/YoloV3_20C)


🔑 Model Architecture:
---------------------
 "YOLOv3": Session10 - Resnet & OCP


🔋 One Cycle Policy: 
-------------------

<p float="left">
  <img src="https://github.com/kishkath/ERA/assets/60026221/c08cfb91-7dd2-4ea4-915f-e3efcea8e292" width = 540 height = 360>
  <img src="https://github.com/kishkath/ERA/assets/60026221/2f8d7bb2-2284-45a5-8000-d652868b5668" width = 540 height = 360>
</p>


💊 Network Results: 
-------------------
 Trained the network for 40 Epochs.
 
 Need to Improve the performance.
 
                    Class accuracy is: 66.972038%
                    No obj accuracy is: 97.798019%
                    Obj accuracy is: 58.063419%
                    100%|██████████| 155/155 [24:44<00:00,  9.58s/it]
                    MAP: 0.23791149258613586


