
import os
import time
from glob import glob

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp
from data import MultiOutputImageDataset
from model import build_unet
from loss import  DiceBCELoss, DiceLoss, FocalLoss
from utils import seeding, create_dir, epoch_time
from torchvision import transforms
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip, Resize, ToTensor
from utils_folder.helpers import Fix_RandomRotation

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

    """ Directories """
    create_dir("files")

    """ Load dataset """
    train_x = sorted(glob("full-data-48x48/train/images/*"))
    train_y = sorted(glob("full-data-48x48/train/orginal/*"))

    valid_x = sorted(glob("full-data-48x48/test/images/*"))
    valid_y = sorted(glob("full-data-48x48/test/orginal/*"))

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print(data_str)

    """ Hyperparameters """
    H = 48
    W = 48
    size = (H, W)
    batch_size = 128
    num_epochs = 500
    lr = 1e-4
    checkpoint_path = "files/orginalModel300E48B_DiceBCELoss.pth"

    transform = transforms.Compose([
            Resize((64, 64)),
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
            ToTensor()
    ])

    transform1 = transforms.Compose([ToTensor()])

    # Define your input and output directories
    input_dir = 'full-data-48x48/train/images'
    output_dir = 'full-data-48x48/train/orginal'

    """ Dataset and loader """
    train_dataset = MultiOutputImageDataset(input_dir, output_dir, transform=transform)
    valid_dataset = MultiOutputImageDataset("full-data-48x48/test/images", "full-data-48x48/test/orginal", transform=transform)

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
    model = build_unet()
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    #loss_fn = DiceBCELoss()
    loss_fn = DiceBCELoss()

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