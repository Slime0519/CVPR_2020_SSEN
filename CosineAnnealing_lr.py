import torch.optim as optim
from cosine_annearing_with_warmup import CosineAnnealingWarmUpRestarts
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    Model = models.vgg19()
    optimizer = optim.Adam(Model.parameters(), lr=0, betas=(0.9, 0.999))
    cosine_scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, T_0=190, T_up=10, T_mult=2, eta_max=1e-4,
                                                     gamma=0.9)

    lr_array = np.zeros(2703)
    for i in range(0,2703):
        #optimizer.step()
        cosine_scheduler.step()

        lr_array[i] = get_lr(optimizer)

    plt.plot(range(0,2703), lr_array)
    plt.show()