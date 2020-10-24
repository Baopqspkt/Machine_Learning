"""
    Author: BaoPham
    Create on: 20/10/2020
    Purpose: detection  and recognize traffic sign 
    Git: https://github.com/Baopqspkt/Machine_Learning branch: master
"""

import torch
import torchvision
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import PIL.Image as Image
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from glob import glob
import shutil
from collections import defaultdict
from torch import nn, optim
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import models
import gdown

class prediction:
    def __init__(self):
        # Select device is cpu
        self.device = "cpu"

        # Define class name for detectioin
        self.class_names = ['priority_road', 'give_way', 'stop', 'no_entry']
        self.DATA_DIR = Path('data')
        self.DATASETS = ['train', 'val', 'test']

        self.mean_nums = [0.485, 0.456, 0.406]
        self.std_nums = [0.229, 0.224, 0.225]

        self.transforms = {'train': T.Compose([
            T.RandomResizedCrop(size=256),
            T.RandomRotation(degrees=15),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(self.mean_nums, self.std_nums)
        ]), 'val': T.Compose([
            T.Resize(size=256),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(self.mean_nums, self.std_nums)
        ]), 'test': T.Compose([
            T.Resize(size=256),
            T.CenterCrop(size=224),
            T.ToTensor(),
            T.Normalize(self.mean_nums, self.std_nums)
        ]),
        }

        self.image_datasets = {
            d: ImageFolder(f'{self.DATA_DIR}/{d}', self.transforms[d]) for d in self.DATASETS
        }

        self.class_names = self.image_datasets['train'].classes
    
        self.base_model = self.create_model(len(self.class_names))
        self.base_model.load_state_dict(torch.load('best_model_state.bin', map_location=torch.device('cpu')))
        self.base_model.eval()

    def create_model(self, n_classes):
        model = models.resnet34(pretrained=False)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, n_classes)
        return model.to(self.device)

    # Covert image to RGB and resize image to 64*64
    def load_image(self, img_path, resize=True):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        return img

    # Show image in new windown
    def show_image(self, img_path):
        img = self.load_image(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    def predict_proba(self, image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = self.transforms['test'](img).unsqueeze(0)

        pred = self.base_model(img.to(self.device))
        pred = F.softmax(pred, dim=1)
        return pred.detach().cpu().numpy().flatten()

    def show_prediction_confidence(self, prediction):
        pred_df = pd.DataFrame({
            'class_names': self.class_names,
            'values': prediction
        })
        sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
        plt.xlim([0, 1])
        plt.show()

def run(path_image_show):

    prediction_sign = prediction()
    prediction_sign.show_image(path_image_show)
    pred = prediction_sign.predict_proba(path_image_show)
    prediction_sign.show_prediction_confidence(pred)

   
if __name__ == '__main__':
    run("data_test/stop-sign-3.jpg")
