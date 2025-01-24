
from glob import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import MultiOutputImageDataset
from model import build_unet
from lite_model import build_lite_unet
from loss import  DiceBCELoss, ComboLoss,CombinedLoss, TverskyLoss, DiceLoss,FocalLoss, FocalTverskyLoss, SoftDiceLoss

from utils import seeding, create_dir
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from train import evaluate
import torch.nn.functional as F

import  time
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

from PIL import Image
from model import build_unet
from utils import create_dir, seeding


# Define the distillation loss
def distillation_loss(y_student, y_teacher, T):
    
    student_probs = torch.sigmoid(y_student)
    teacher_probs = torch.sigmoid(y_teacher / T)

    eps = 1e-7
    student_probs = torch.clamp(student_probs, eps, 1 - eps)
    teacher_probs = torch.clamp(teacher_probs, eps, 1 - eps)


    kl_div = teacher_probs * torch.log(teacher_probs / student_probs) + \
             (1 - teacher_probs) * torch.log((1 - teacher_probs) / (1 - student_probs))
    distillation_loss1 = kl_div.mean()

    
    return distillation_loss1 * (T ** 2)

# Train the student model with knowledge distillation
def train_with_kd(student_model, teacher_models, trainloader, valid_loader, loss_fn, optimizer, num_epochs=5, temperature=3.0, alpha=0.7, device=None, loss_name='', scheduler = None):
    student_model.to(device)
    best_valid_loss = float("inf")
    for epoch in range(num_epochs):
        student_model.train()
        running_loss = []
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs_student = student_model(inputs)
            outputs_teacher = [model(inputs) for model in teacher_models]
            losses_kd = [distillation_loss(outputs_student, output_teacher, temperature) for output_teacher in outputs_teacher]
            loss_kd = sum(losses_kd) / len(teacher_models)
            loss_ce = loss_fn(outputs_student, labels)
            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce           
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            """
            if i % 100 == 99:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(trainloader)}], Loss: {running_loss / 100:.4f}')
                running_loss = 0.0
            """
        scheduler.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {sum(running_loss) / len(running_loss):.4f}')
        valid_loss = evaluate(student_model, valid_loader, loss_fn, device)
        if valid_loss < best_valid_loss:
            data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}."
            print(data_str)
            best_valid_loss = valid_loss
                                                     
            torch.save(student_model.state_dict(), f"files/T123_UNET_DistillationToStudent_CHASEDB1.pth")
    print('Finished Training with KD')



if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    batch_size = 1
    num_epochs = 250
    lr = 1e-3

    """ Load the checkpoint """
    device = torch.device('cuda:0') 
    teacher1 = build_unet()
    teacher1 = teacher1.to(device)
    teacher1.load_state_dict(torch.load("files/T1_CHASEDB1_originalSize_CombinedLoss.pth", map_location=device))
    teacher1.eval()

    teacher2 = build_unet()
    teacher2 = teacher2.to(device)
    teacher2.load_state_dict(torch.load("files/T2_CHASEDB1_ThickOrjSize_CombinedLoss.pth", map_location=device))
    teacher2.eval()

    teacher3 = build_unet()
    teacher3 = teacher3.to(device)
    teacher3.load_state_dict(torch.load("files/T3_CHASEDB1_ThinOrjSize_CombinedLoss.pth", map_location=device))
    teacher3.eval()

    transform = transforms.Compose([
        transforms.ToTensor()          # Convert images to PyTorch tensors
    ])

    # Define your input and output directories
    train_input_dir = 'C:/Users/FSM/Desktop/MUSA/KD_CHASEDB1/data/input/train'
    train_target_dir = 'C:/Users/FSM/Desktop/MUSA//KD_CHASEDB1/data/target/train/orj'

    val_input_dir = 'C:/Users/FSM/Desktop/MUSA/KD_CHASEDB1/data/input/val'
    val_target_dir = 'C:/Users/FSM/Desktop/MUSA/KD_CHASEDB1/data/target/val/orj'

   
 

    """ Dataset and loader """
    train_dataset = MultiOutputImageDataset(train_input_dir, train_target_dir, transform=transform, type='orj')
    valid_dataset = MultiOutputImageDataset(val_input_dir, val_target_dir, transform=transform, type='orj')
    
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

    teacher_models = [teacher1, teacher2, teacher3]
 
    loss_functions = {    
        "FocalTverskyLoss":FocalTverskyLoss(),
        "CombinedLoss": CombinedLoss(),
        "TverskyLoss":TverskyLoss(),        
        "DiceLoss": DiceLoss(),
        "ComboLoss":ComboLoss(),
        "SoftDiceLoss":SoftDiceLoss(),
        "DiceBCELoss": DiceBCELoss(),
        "FocalLoss": FocalLoss()
    }
    
    # # Training loop for each loss function
    for loss_name, loss_fn in loss_functions.items():
        student_model = build_unet()
        student_model = student_model.to(device)
        optimizer = optim.Adam(student_model.parameters(), lr=0.001, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

        print(f"Training with {loss_name}")
        train_with_kd(student_model, teacher_models, train_loader, valid_loader, loss_fn, optimizer, num_epochs=num_epochs, temperature=3.0, alpha=0.5, device=device, loss_name = loss_name, scheduler = scheduler)
    







    