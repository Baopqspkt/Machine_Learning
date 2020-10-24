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


def run():
    torch.multiprocessing.freeze_support()
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    def load_image(img_path, resize=True):
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        if resize:
            img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        return img

    def show_image(img_path):
        img = load_image(img_path)
        plt.imshow(img)
        plt.axis('off')
        plt.show()

    class_names = ['priority_road', 'give_way', 'stop', 'no_entry']
    DATA_DIR = Path('data')
    DATASETS = ['train', 'val', 'test']

    mean_nums = [0.485, 0.456, 0.406]
    std_nums = [0.229, 0.224, 0.225]

    transforms = {'train': T.Compose([
        T.RandomResizedCrop(size=256),
        T.RandomRotation(degrees=15),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]), 'val': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]), 'test': T.Compose([
        T.Resize(size=256),
        T.CenterCrop(size=224),
        T.ToTensor(),
        T.Normalize(mean_nums, std_nums)
    ]),
    }

    image_datasets = {
        d: ImageFolder(f'{DATA_DIR}/{d}', transforms[d]) for d in DATASETS
    }

    class_names = image_datasets['train'].classes
    print("CLASS_NAMES: ",class_names)

    def create_model(n_classes):
        model = models.resnet34(pretrained=False)
        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, n_classes)
        return model.to(device)

    base_model = create_model(len(class_names))
    base_model.load_state_dict(torch.load('best_model_state.bin', map_location=torch.device('cpu')))
    base_model.eval()

    def predict_proba(model, image_path):
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = transforms['test'](img).unsqueeze(0)

        pred = model(img.to(device))
        pred = F.softmax(pred, dim=1)
        return pred.detach().cpu().numpy().flatten()

    def show_prediction_confidence(prediction, class_names):
        pred_df = pd.DataFrame({
            'class_names': class_names,
            'values': prediction
        })
        sns.barplot(x='values', y='class_names', data=pred_df, orient='h')
        plt.xlim([0, 1])
        plt.show()

    show_image('stop-sign-1.jpg')
    pred = predict_proba(base_model, 'stop-sign-1.jpg')
    print(pred)
    show_prediction_confidence(pred, class_names)

   
if __name__ == '__main__':
    run()
