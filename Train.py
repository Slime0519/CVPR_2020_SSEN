import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader

from Data_gen import Dataset_Vaild, Dataset_Train
from Baseline import Baseline, L1_Charbonnier_loss

import argparse
import numpy as np
import os

parser = argparse.ArgumentParser(description="RefSR Network with SSEN Training module")
parser.add_argument('--pre_trained', type = str, default=None, help = "path of pretrained modules")
parser.add_argument('--num_epochs', type = int, default = 1000000, help = "Number of epochs")
parser.add_argument('--pre_resulted', type = str, default = None, help = "Data array of previous step")
parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size")
parser.add_argument('--learning_rate', type = float, default=1e-4, help ="learning rate")
parser.add_argument('--gamma', type = float, default = 0.9, help = 'momentum of ADAM optimizer')

if __name__ == "__main__":
    opt = parser.parse_args()

    TOTAL_EPOCHS = opt.num_epochs
    PRETRAINED_PATH = opt.pre_trained
    PRERESULTED_PATH = opt.pre_resulted
    BATCH_SIZE = opt.batch_size
    lr = opt.learning_rate
    gamma = opt.gamma

    TrainDIR_PATH = "CUFED_SRNTT/input/"
    RefDIR_PATH = "CUFED_SRNTT/ref/"
    VaildDIR_PATH = "CUFED_SRNTT/CUFED5/"

    ResultSave_PATH = "Result_metrics"
    TrainedMODEL_PATH = "Trained_model"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Train_Dataset = Dataset_Train(dirpath_input=TrainDIR_PATH, dirpath_ref=RefDIR_PATH, upscale_factor=4)
    Vaild_Dataset = Dataset_Vaild(dirpath=VaildDIR_PATH, upscale_factor=4)

    Train_Dataloader = DataLoader(dataset=Train_Dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)
    Vaild_Dataloader = DataLoader(dataset=Vaild_Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)

    Model = Baseline()
    Model = Model.to(device)
    optimizer = optim.Adam(Model.parameters(), lr=lr, betas=(0.9, 0.999))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, )

    criterion = L1_Charbonnier_loss().to(device)
    MSELoss_criterion = nn.MSELoss()
    loss_array_Train = np.zeros(TOTAL_EPOCHS)
    PSNR_array_Train = np.zeros(TOTAL_EPOCHS)
    PSNR_array_Vaild = np.zeros(TOTAL_EPOCHS)

    trainloader_len = len(Train_Dataloader)

    for epoch in range(TOTAL_EPOCHS):
        Model.train()
        avg_PSNR = 0
        avg_loss = 0
        print("----Training step-----")
        for i,(lr_image, hr_image, ref_image) in enumerate(Train_Dataloader):
            lr_image, hr_image, ref_image = lr_image.to(device), hr_image.to(device), ref_image.to(device)
            optimizer.zero_grad()

            sr_image = Model(lr_image, ref_image)

            loss = criterion(sr_image, hr_image)
            avg_loss += loss

            MSELoss = MSELoss_criterion(sr_image, hr_image)
            avg_PSNR += 10 * torch.log10(1/MSELoss)

            loss.backward()
            optimizer.step()
            print("epoch {} training step : {}/{}".format(epoch + 1, i + 1, trainloader_len))

        cosine_scheduler.step()

        PSNR_array_Train[epoch] = avg_PSNR/len(Train_Dataloader)
        loss_array_Train[epoch] = loss/len(Train_Dataloader)

        print("Training average PSNR : {}, loss : {}".format(PSNR_array_Train[epoch], loss_array_Train[epoch]))

        Model.eval()
        avg_PSNR = 0
        print("----Evaluation Step----")

        with torch.no_grad():
            for lr_image, hr_image in Vaild_Dataloader:
                lr_image = lr_image.to(device)
                sr_image = Model(lr_image, ref_image)

                MSELoss = MSELoss_criterion(sr_image,hr_image)
                avg_PSNR += 10*torch.log10(1/MSELoss)

            PSNR_array_Vaild[epoch] = avg_PSNR/len(Vaild_Dataloader)
            print("evaluation average PSNR : {}".format(PSNR_array_Vaild[epoch]))

        if (epoch+1) % 1000 == 0:
            np.save(os.path.join(ResultSave_PATH,"Training_Average_PSNR.npy"),PSNR_array_Train)
            np.save(os.path.join(ResultSave_PATH,"Training_Average_loss.npy"),loss_array_Train)
            np.save(os.path.join(ResultSave_PATH,"Vaild_Average_PSNR.npy"),PSNR_array_Vaild)

            torch.save(Model.state_dict(), os.path.join(TrainedMODEL_PATH,"Model_epoch{}.pth".format(epoch+1)))

