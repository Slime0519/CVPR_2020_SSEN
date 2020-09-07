import ray
from ray.util.sgd import TorchTrainer

import torch
import torch.optim as optim

from torch.utils.data import DataLoader

from Data_gen import Dataset_Vaild, Dataset_Train
from Models.Train.Baseline import Baseline, L1_Charbonnier_loss
from Models.Train.Baseline_big import BigBaseline

import argparse
from tqdm import trange

parser = argparse.ArgumentParser(description="RefSR Network with SSEN Training module")
parser.add_argument('--pre_trained', type = str, default=None, help = "path of pretrained modules")
parser.add_argument('--num_epochs', type = int, default = 1000000, help = "Number of epochs")
parser.add_argument('--pre_resulted', type = str, default = None, help = "Data array of previous step")
parser.add_argument('--batch_size', type = int, default = 32, help = "Batch size")
parser.add_argument('--learning_rate', type = float, default=1e-4, help ="learning rate")
parser.add_argument('--gamma', type = float, default = 0.9, help = 'momentum of ADAM optimizer')
parser.add_argument('--pretrained_epoch', type=int, default=0, help ='pretrained models epoch')

TrainDIR_PATH = "CUFED_SRNTT/input/"
RefDIR_PATH = "CUFED_SRNTT/ref/"
VaildDIR_PATH = "CUFED_SRNTT/CUFED5/"

def model_creator(config):
    args = config["args"]
    if config['model_category'] == "big":
        model = BigBaseline()
    else:
        model = Baseline()

    if config["num_workers"] > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    return model

def optimizer_creator(model, config):
    return optim.Adam(model.parameters(), lr=config["lr"], betas=config["betas"])

def data_creator(config):
    Train_Dataset = Dataset_Train(dirpath_input=TrainDIR_PATH, dirpath_ref=RefDIR_PATH, upscale_factor=4)
    Vaild_Dataset = Dataset_Vaild(dirpath=VaildDIR_PATH, upscale_factor=4)

    Train_Dataloader = DataLoader(dataset=Train_Dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=8,drop_last=True, pin_memory=True)
    Vaild_Dataloader = DataLoader(dataset=Vaild_Dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=8, drop_last=True,pin_memory=True)
    return Train_Dataloader, Vaild_Dataloader

def loss_creator(config):
    return L1_Charbonnier_loss()

def scheduler_creator(optimizer,config):
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_EPOCHS, )
    return cosine_scheduler

if __name__ == "__main__":
    opt = parser.parse_args()

    TOTAL_EPOCHS = opt.num_epochs
    PRETRAINED_PATH = opt.pre_trained
    PRERESULTED_PATH = opt.pre_resulted
    BATCH_SIZE = opt.batch_size
    lr = opt.learning_rate
    gamma = opt.gamma
    PRETRAINED_EPOCH = opt.pretrained_epoch

    ResultSave_PATH = "Result_metrics"
    TrainedMODEL_PATH = "Trained_model"

    ray.init()
    trainer = TorchTrainer(model_creator=model_creator,
                           data_creator=data_creator,
                           optimizer_creator=optimizer_creator,
                           loss_creator=loss_creator,
                           scheduler_creator=scheduler_creator,
                           use_gpu = True,
                           config = {"model_category":"small", "lr" : lr , "betas":(0.9,0.999),"num_workers":4, "BATCH_SIZE":BATCH_SIZE},
                           use_tqdm=True,
                           use_fp16=True,
                           scheduler_step_freq='epoch'
                           )

    pbar = trange(PRETRAINED_EPOCH,TOTAL_EPOCHS, unit="epoch")
    for i in pbar:
        # Increase `max_retries` to turn on fault tolerance.
        trainer.train()
       # val_stats = trainer.validate()
      #  pbar.set_postfix(dict(acc=val_stats["val_accuracy"]))


    print(trainer.validate())
    trainer.shutdown()
    print("success!")