"""
Main file for training Yolo model on Pascal VOC and COCO dataset
"""

import pytorch_lightning as pl
import config
import torch
import torch.optim as optim

from model import YOLOv3
from tqdm import tqdm
from utils import (
    mean_average_precision,
    cells_to_bboxes,
    get_evaluation_bboxes,
    save_checkpoint,
    load_checkpoint,
    check_class_accuracy,
    plot_couple_examples
)

from torch.cuda.amp import autocast, GradScaler
from loss import YoloLoss
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True

class YOLOv3_Arch(pl.LightningModule):
    def __init__(self,scaled_anchors):
        super().__init__()
        self.loss_fn = YoloLoss()
        self.scaler = torch.cuda.amp.GradScaler()
        self.scaled_anchors = scaled_anchors
        self.model = YOLOv3()
        
    def forward(self,x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                            steps_per_epoch=len(self.train_dataloader()),
                                                            epochs=20,pct_start=5/20,div_factor=100,
                                                             three_phase=False,final_div_factor=100,
                                                             anneal_strategy='linear'),
  
            'monitor': 'val_loss'
        }
        return [optimizer],[lr_scheduler]

    def training_step(self,batch,batch_idx):# scaler, scaled_anchors):
        x,y = batch
        y0,y1,y2 = y
        with autocast():
            out = self.model(x)
            loss = (
                self.loss_fn(out[0], y0, self.scaled_anchors[0])
                + self.loss_fn(out[1], y1, self.scaled_anchors[1])
                + self.loss_fn(out[2], y2, self.scaled_anchors[2])
            )

        self.log('train_loss',loss,prog_bar=True,sync_dist=True)
        return loss
    

    def on_train_end(self):
        
        plot_couple_examples(self.model, self.test_dataloader(), 0.6, 0.5, self.scaled_anchors)
        check_class_accuracy(self.model, self.train_dataloader(), threshold=config.CONF_THRESHOLD)
        
        pred_boxes, true_boxes = get_evaluation_bboxes(
            self.test_dataloader(),
            self.model,
            iou_threshold=config.NMS_IOU_THRESH,
            anchors=config.ANCHORS,
            threshold=config.CONF_THRESHOLD,
        )
        mapval = mean_average_precision(
            pred_boxes,
            true_boxes,
            iou_threshold=config.MAP_IOU_THRESH,
            box_format="midpoint",
            num_classes=config.NUM_CLASSES,
        )
        
        print(f"MAP: {mapval.item()}")

    def train_dataloader(self):
        train_path = "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/train.csv"
        test_path = "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/test.csv"
        trainloader = get_loadings(train_csv_path=train_path,test_csv_path=test_path)[0]
        return trainloader

    def val_dataloader(self):
        train_path = "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/train.csv"
        test_path = "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/test.csv"
        valloader = get_loadings(train_csv_path=train_path,test_csv_path=test_path)
        return valloader[2]

    def test_dataloader(self):
        train_path = "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/train.csv"
        test_path = "/kaggle/input/pascal-voc-dataset-used-in-yolov3-video/PASCAL_VOC/test.csv"
        testloader = get_loadings(train_csv_path=train_path,test_csv_path=test_path)
        return testloader[1]


scaled_anchors = (
    torch.tensor(config.ANCHORS)
    * torch.tensor(config.S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
).to("cuda")   

yolo_model = YOLOv3_Arch(scaled_anchors)

checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/',filename='best_weights',monitor='val_loss',save_top_k=1)

trainer = Trainer(
    max_epochs=40,
    accelerator="gpu",
    devices=1,  # limiting got iPython runs
    callbacks=[checkpoint_callback],
)
trainer.fit(yolo_model)
