import matplotlib.pyplot as plt
import numpy as np
import os

ResultSave_PATH = "Result_metrics"

def get_length(array):
    i = 0
    while array[i]!=0 and i!=len(array):
        i += 1
    return i

def plot_chart(array, title, xlabel, ylabel, save = False):
    length = get_length(array)
    y_max = np.max(array)
    y_max += y_max//5 #plus margin about y axis

    x = range(length)
    y = range(y_max)

    plt.plot(x,y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save :
        plt.savefig(os.path.join(ResultSave_PATH,title+".png"),dpi = 500)
    else :
        plt.show()


if __name__ == "__main__":
    Train_PSNR = np.load(os.path.join(ResultSave_PATH,"Training_Average_PSNR.npy"))
    Train_loss = np.load(os.path.join(ResultSave_PATH,"Training_Average_loss.npy"))
    Eval_Average_PSNR = np.load(os.path.join(ResultSave_PATH,"Vaild_Average_PSNR.npy"))

    plot_chart(Train_PSNR,"Average PSNR in Training step","epochs","PSNR",save = False)
    plot_chart(Train_loss, "Average loss in Training step","epochs", "loss",save=False)
    plot_chart(Eval_Average_PSNR,"Average PSNR in evaluation step","epochs","PSNR",save=False)



