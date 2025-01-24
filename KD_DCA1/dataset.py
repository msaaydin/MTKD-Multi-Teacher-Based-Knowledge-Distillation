import os
import pickle
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, ToTensor
from utils_folder.helpers import Fix_RandomRotation
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
class ZeroPadTransform:
  def __init__(self, pad_width):
      self.pad_width = pad_width

  def __call__(self, tensor):
      return F.pad(tensor, self.pad_width, mode='constant', value=0)
transformsT =transforms.ToTensor()

class vessel_dataset(Dataset):
    def __init__(self, path, mode, is_val=False, split=None, de_train=False):

        self.mode = mode
        self.is_val = is_val
        self.de_train = de_train
        self.data_path = os.path.join(path, f"{mode}")
        self.data_file = os.listdir(self.data_path+'/gt')

        self.img_file = self._select_img(self.data_file)
        if split is not None and mode == "training":
            assert split > 0 and split < 1
            if not is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))]
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):]

        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
            ToTensor()
        ])

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        img_pt =os.path.join(self.data_path,'img', img_file)
        img = Image.open(img_pt)
        img = TF.pad(img, (0, 0, 20, 20))
        # with open(, mode='rb') as file:
        #     img = torch.from_numpy(pickle.load(file)).float()


        gt_file = os.path.join(self.data_path,'gt', img_file)
        gt = Image.open(gt_file)
        gt = TF.pad(gt, (0, 0, 20, 20))
        # with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
        #     gt = torch.from_numpy(pickle.load(file)).float()

        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        return transformsT(img), transformsT(gt)

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            # if file[:3] == "img":
            img_list.append(file)

        return img_list

    def __len__(self):
        if self.de_train == False:
            return len(self.img_file)
        else:
            return len(self.img_file) // 2

    def readIndexes(self, path):
        lines = []
        with open(f"{path}", "r") as f:
            lines = f.readlines()
        
        lines = [fname.replace("\n","") for fname in lines]
        return lines
