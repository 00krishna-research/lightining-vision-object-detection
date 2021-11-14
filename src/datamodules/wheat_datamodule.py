
from typing import Optional, Tuple

import math
import sys
import time
from tqdm.notebook import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Torch imports 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms as transforms
from torchvision.ops.boxes import box_iou
from torchvision.models.detection._utils import Matcher
from torchvision.ops import nms, box_convert

# Albumentations is used for the Data Augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Pytorch import
from pytorch_lightning import seed_everything
from pytorch_lightning import LightningDataModule

default_transforms = A.Compose([
    A.LongestMaxSize(1024,p=1),
    A.PadIfNeeded(min_height=1024,min_width=1024,p=1,border_mode=1,value=0),
    A.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
    ),
    ToTensorV2(),
],bbox_params=A.BboxParams(format='pascal_voc',label_fields=['class_labels'],min_area=20))


class WheatDataset(Dataset):
    """A dataset example for GWC 2021 competition."""

    def __init__(self, annotations, root_dir, transforms=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional data augmentation to be applied
                on a sample.
        """

        self.root_dir = Path(root_dir)
        self.image_list = annotations["image_name"].values
        self.domain_list = annotations["domain"].values
        self.boxes = [self.decodeString(item) for item in annotations["BoxesString"]]        
        self.transforms = transforms

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        
        imgp = str(self.root_dir / (self.image_list[idx]+".png"))
        domain = self.domain_list[idx] # We don't use the domain information but you could !
        bboxes = self.boxes[idx]
        img = cv2.imread(imgp)
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Opencv open images in BGR mode by default

        if self.transforms:
            transformed = self.transforms(image=image,
                                         bboxes=bboxes,
                                         class_labels=["wheat_head"]*len(bboxes)) #Albumentations can transform images and boxes
            image = transformed["image"]
            bboxes = transformed["bboxes"]

        if len(bboxes) > 0:
          bboxes = torch.stack([torch.tensor(item) for item in bboxes])
        else:
          bboxes = torch.zeros((0,4))
        return image, bboxes, domain


    def decodeString(self,BoxesString):
      """
      Small method to decode the BoxesString
      """
      if BoxesString == "no_box":
          return np.zeros((0,4))
      else:
          try:
              boxes =  np.array([np.array([int(i) for i in box.split(" ")])
                              for box in BoxesString.split(";")])
              return boxes
          except:
              print(BoxesString)
              print("Submission is not well formatted. empty boxes will be returned")
              return np.zeros((0,4))



class WheatDetDataModule(LightningDataModule):
    
    def __init__(self,
                root_dir, csv_file,
                transforms=default_transforms,
                num_workers=4,
                batch_size=8, 
                test_percentage=0.20, 
                shuffle=True, 
                pin_memory=True):
        
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.full_image_annotations = pd.read_csv(csv_file)
        self.transforms = transforms
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.test_percentage = test_percentage
        self.shuffle = shuffle
        self.pin_memory=pin_memory

    def setup(self, stage: Optional[str] = None):
        traindf, testdf = train_test_split(self.full_image_annotations, 
                                        test_size=self.test_percentage,
                                        shuffle=self.shuffle)

        traindf, valdf = train_test_split(traindf, 
                                      test_size=self.test_percentage,
                                      shuffle=self.shuffle)

        self.trainds = WheatDataset(traindf, self.root_dir, transforms=self.transforms)
        self.valds = WheatDataset(valdf, self.root_dir, transforms=self.transforms)
        self.testds = WheatDataset(testdf, self.root_dir, transforms=self.transforms)

    def train_dataloader(self) -> DataLoader:
        train_loader = DataLoader(
            self.trainds,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader


    def val_dataloader(self) -> DataLoader:
        valid_loader = DataLoader(
            self.valds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return valid_loader

    def test_dataloader(self) -> DataLoader:

        test_loader = DataLoader(
            self.testds,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            drop_last=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )
        return test_loader

    @staticmethod
    def collate_fn(batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images = list()
        targets=list()
        metadatas = list()

        for i, t, m in batch:
            images.append(i)
            targets.append(t)
            metadatas.append(m)
        images = torch.stack(images, dim=0)

        return images, targets,metadatas














