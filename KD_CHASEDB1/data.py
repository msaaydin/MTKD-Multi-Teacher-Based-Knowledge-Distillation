import os
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from glob import glob
class MultiOutputImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None, type='orj'):
        self.input_dir = input_dir
        self.transform = transform

        # Get all input images
        self.input_images = glob( input_dir + "/*.jpg", recursive=True)
        self.target_path = target_dir
        
        self.type = type

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load input image
        # Load input image
        if self.type == 'orj':
            nm = os.path.basename(self.input_images[idx])[9:]
            trg = str.replace(str.replace(os.path.basename(self.input_images[idx]),nm,'_1stHO'+nm), '.jpg','.png')
            target_image_path = os.path.join(self.target_path,trg)
        elif self.type == 'thick':
            nm = os.path.basename(self.input_images[idx])[9:]
            trg = str.replace(str.replace(os.path.basename(self.input_images[idx]),nm,'_1stHO_thick'+nm), '.jpg','.png')
            
            target_image_path = os.path.join(self.target_path,trg)
        elif self.type == 'thin':
            nm = os.path.basename(self.input_images[idx])[9:]
            trg = str.replace(str.replace(os.path.basename(self.input_images[idx]),nm,'_1stHO_thin'+nm), '.jpg','.png')
            
            target_image_path = os.path.join(self.target_path,trg)
        input_image_path =  self.input_images[idx]
        # input_image = Image.open(input_image_path).convert("RGB")
        input_image = Image.open(input_image_path)
        # Load target image and convert it to grayscale (1 channel)
        # target_path = os.path.join(self.output_dir, self.input_images[idx])
        out_im = Image.open(target_image_path).convert("L")  # Convert to grayscale (1 channel)
        input_image = TF.pad(input_image, (0, 0, 9, 48))
        out_im = TF.pad(out_im, (0, 0, 9, 48))
        # Apply transforms if provided
        if self.transform:
            input_image = self.transform(input_image)
            out_im = self.transform(out_im)
        
        return input_image, out_im
