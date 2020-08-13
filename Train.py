import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Data_gen import Dataset_Vaild, Dataset_Train
from Baseline import Baseline, L1_Charbonnier_loss

BATCH_SIZE = 32
TOTAL_EPOCHS = 100000
lr = 1e-4
gamma = 0.9

if __name__ == "__main__":

    TrainDIR_PATH = "input/"
    RefDIR_PATH = "ref/"
    VaildDIR_PATH = "CUFED5/"

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    Train_Dataset = Dataset_Train(dirpath_input=TrainDIR_PATH, dirpath_ref=RefDIR_PATH, upscale_factor=4)
    Vaild_Dataset = Dataset_Vaild(dirpath=VaildDIR_PATH, upscale_factor=4)

    Train_Dataloader = DataLoader(dataset=Train_Dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    Vaild_Dataloader = DataLoader(dataset=Vaild_Dataset, batch_size=1, shuffle=False, num_workers=0)

    Model = Baseline()
    Model = Model.to(device)
    optimizer = optim.Adam(Model.parameters(), lr=lr, betas=(0.9, 0.999))
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, )

    criterion = L1_Charbonnier_loss().to(device)

    for epoch in range(TOTAL_EPOCHS):
        Model.train()

        for lr_image, hr_image, ref_image in Train_Dataloader:
            lr_image, hr_image, ref_image = lr_image.to(device), hr_image.to(device), ref_image.to(device)
            optimizer.zero_grad()

            sr_image = Model(lr_image, ref_image)
            loss = criterion(sr_image, hr_image)

            loss.backward()
            optimizer.step()

        cosine_scheduler.step()

        Model.eval()

        with torch.no_grad():
            for lr_image, hr_image in Vaild_Dataloader:
                lr_image = lr_image.to(device)
                sr_image = Model(lr_image, ref_image)


