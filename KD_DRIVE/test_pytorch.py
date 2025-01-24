import torch
import pandas as pd
from tester import Tester
from model import build_unet
from torch.utils.data import DataLoader
from dataset import vessel_dataset
from lite_model import build_lite_unet

def main(path):
    device = torch.device('cuda:1')
    result_df = pd.DataFrame(columns=['DATASET', 'ENCODER_NAME', 'SEED', 'AUC',	'F1', 'Acc', 'Sen', 'Spe', 'pre', 'IOU'])
    for data_flag in ['DRIVE']:
        print(data_flag)
        model = build_unet()                                                                                                                                                       
        model.load_state_dict(torch.load(f"files/T123_UNET_DistillationToStudent_DRIVE.pth",map_location=device))
        model.to(device)
        pt = "C:/Users/FSM/Desktop/MUSA/KD_DRIVE/data"
        test_dataset = vessel_dataset(pt, mode="test")
        test_loader = DataLoader(test_dataset, 1,
                            shuffle=False, pin_memory=True)

        test = Tester(model, test_loader, pt, path, None, show=True, data_flag = data_flag)
        result_df = test.test(result_df, data_flag, 'unet', None)
    
        result_df.to_excel(f"files/T123_UNET_DistillationToStudent_DRIVE.xlsx")

if __name__ == '__main__':
    main('results')