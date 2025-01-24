
import os
import time
from glob import glob
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision.transforms.functional as TF

import segmentation_models_pytorch as smp
from data import MultiOutputImageDataset

from lite_model import build_lite_unet
from loss import  DiceBCELoss, DiceLoss, FocalLoss, CombinedLoss
from utils import seeding, create_dir, epoch_time
from torchvision import transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor
from utils_folder.helpers import Fix_RandomRotation
from torch.utils.data import Dataset
from PIL import Image

class MyDataSet(Dataset):
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
        if self.type == 'orj':
            target_image_path = os.path.join(self.target_path,os.path.basename(str.replace(self.input_images[idx],'.jpg','_1stHO.png')))
        elif self.type == 'thick':
            target_image_path = os.path.join(self.target_path,os.path.basename(str.replace(self.input_images[idx],'.jpg','_1stHO_thick.png')))
        elif self.type == 'thin':
            target_image_path = os.path.join(self.target_path,os.path.basename(str.replace(self.input_images[idx],'.jpg','_1stHO_thin.png')))
        input_image_path =  self.input_images[idx]
        input_image = Image.open(input_image_path)
        out_im = Image.open(target_image_path).convert("L")  # Convert to grayscale (1 channel)
        input_image = TF.pad(input_image, (0, 0, 9, 48))
        out_im = TF.pad(out_im, (0, 0, 9, 48))
        # Apply transforms if provided
        if self.transform:
            input_image = self.transform(input_image)
            out_im = self.transform(out_im)
        
        return input_image, out_im




def train(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0

    model.train()
    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss

def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

        epoch_loss = epoch_loss/len(loader)
    return epoch_loss

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Hyperparameters """
   
    batch_size = 10
    num_epochs = 110
    lr = 1e-3
    checkpoint_path = "files/f_Lite_Unet_NO_KD/Lite_Unet_NO_KD__CHASEDB1.pth"

    transform1 = transforms.Compose([ToTensor()])

    # Define your input and output directories
    train_input_dir = 'C:/Users/FSM/Desktop/MUSA/KD_CHASEDB1/data/input/train'
    train_target_dir = 'C:/Users/FSM/Desktop/MUSA//KD_CHASEDB1/data/target/train/orj'

    val_input_dir = 'C:/Users/FSM/Desktop/MUSA/KD_CHASEDB1/data/input/val'
    val_target_dir = 'C:/Users/FSM/Desktop/MUSA/KD_CHASEDB1/data/target/val/orj'

    """ Dataset and loader """
    train_dataset = MyDataSet(train_input_dir, train_target_dir, transform=transform1, type='orj')
    valid_dataset = MyDataSet(val_input_dir, val_target_dir, transform=transform1, type='orj')

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device('cuda:1')   
    model = build_lite_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #loss_fn = DiceBCELoss()
    loss_fn = CombinedLoss()

    """ Training the model """
    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        """ Saving the model """
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {checkpoint_path}"
            print(data_str)

            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print(data_str)
print("ggg")