import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from glob import glob
class MultiOutputImageDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform

        # Get all input images
        self.input_images = glob( input_dir + "/*.png", recursive=True)
        self.target_path = target_dir

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load input image
        target_image_path = os.path.join(self.target_path,os.path.basename(self.input_images[idx]))
        input_image_path =  self.input_images[idx]
        input_image = Image.open(input_image_path).convert('L')
        out_im = Image.open(target_image_path).convert("L")  # Convert to grayscale (1 channel)
        
        # Apply transforms if provided
        if self.transform:
            input_image = self.transform(input_image)
            out_im = self.transform(out_im)
        
        return input_image, out_im
