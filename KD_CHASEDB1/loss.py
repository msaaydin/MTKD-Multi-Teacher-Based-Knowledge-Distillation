import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # Apply sigmoid if your model doesn't have an activation layer
        inputs = torch.sigmoid(inputs)


        # Resize inputs to match targets' size
        if inputs.shape != targets.shape:
            inputs = torch.nn.functional.interpolate(inputs, size=targets.shape[2:], mode='bilinear', align_corners=False)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        # Binary Cross-Entropy Loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Final Dice + BCE loss
        Dice_BCE = BCE + dice_loss

        return Dice_BCE
    

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice


    
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, weight=None, size_average=True):
        """
        Focal Loss for binary classification.
        
        Parameters:
            alpha (float): Balance factor between classes. Default is 1.
            gamma (float): Focusing parameter to reduce the relative loss for easy examples. Default is 2.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        

    def forward(self, inputs, targets):
        # Apply sigmoid activation if not included in the model
        inputs = torch.sigmoid(inputs)

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Calculate the binary cross-entropy loss
        bce_loss = nn.BCELoss(reduction='none')(inputs, targets)
        
        # Calculate focal loss
        pt = torch.exp(-bce_loss)  # pt is the probability of the correct class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss
        
        return focal_loss.mean()

class DiceLoss2(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss2, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid if not already included
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
 
class RegionWeightedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1):
        super(RegionWeightedLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, inputs, targets, weight_map):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid if not included in the model
        
        # Flatten tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weight_map = weight_map.view(-1)

        # Dice Loss
        intersection = (inputs * targets * weight_map).sum()
        union = ((inputs + targets) * weight_map).sum()
        dice_loss = 1 - (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return dice_loss



class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1, focal_alpha=0.8, focal_gamma=2):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.dice_loss = DiceLoss2(smooth=smooth)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    def forward(self, inputs, targets):
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)
        combined = self.alpha * dice + (1 - self.alpha) * focal
        return combined


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # Apply sigmoid if not already included in the model
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Compute true positives, false negatives, and false positives
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return 1 - Tversky
    



class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=0.75, smooth=1):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - Tversky) ** self.gamma



class ComboLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, smooth=1):
        super(ComboLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # BCE Loss
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        # Focal Loss
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = (1 - pt) ** self.gamma * F.binary_cross_entropy(inputs, targets, reduction='none')
        focal_loss = focal_loss.mean()

        # Combo Loss
        return self.alpha * BCE + (1 - self.alpha) * dice_loss + focal_loss

class ComboLoss2(nn.Module):
    def __init__(self, alpha=0.5, smooth=1):
        super(ComboLoss2, self).__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Dice Loss
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # BCE Loss
        BCE = nn.functional.binary_cross_entropy(inputs, targets, reduction='mean')

        # Combine both losses
        return self.alpha * BCE + (1 - self.alpha) * dice_loss


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

