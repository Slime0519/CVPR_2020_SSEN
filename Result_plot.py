import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

ResultSave_PATH = "Result_metrics"

parser = argparse.ArgumentParser(description="RefSR Network with SSEN Training module")
parser.add_argument('--model_size', type = str, default="normal128", help = "select model size")

def get_length(array):
    i = 0
    while i!=len(array) and array[i]!=0:
        i += 1
    return i

def plot_chart(array, title, xlabel, ylabel, save = False):
    length = get_length(array)
    y_max = np.max(array)
    y_max += y_max//5 #plus margin about y axis
    print(length)
    x = range(length)
   # x = range(1000)
    y = range(int(y_max))

    plt.plot(x,array[:length])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if save :
        plt.savefig(os.path.join(ResultSave_PATH,title+".png"),dpi = 500)
    else :
        plt.show()


if __name__ == "__main__":
    opt = parser.parse_args()

    Modelsize = opt.model_size

    if Modelsize == "normal":
        prefix_resultname = "normalModel_add"
    elif Modelsize == "normal_concat":
        prefix_resultname = "normalModel_concat"
    elif Modelsize == "normal_concat_cosine":
        prefix_resultname = "normalModel_cosine_concat"
    elif Modelsize == "normal128":
        prefix_resultname = "normalModel_model128"
    elif Modelsize == "big":
        prefix_resultname = "bigModel"
    elif Modelsize == 'normal_light':
        prefix_resultname = "normalModel_light"
    else:
        prefix_resultname = "smallModel"

    Train_PSNR = np.load(os.path.join(ResultSave_PATH,prefix_resultname+"_Training_Average_PSNR.npy"))
    Train_loss = np.load(os.path.join(ResultSave_PATH,prefix_resultname+"_Training_Average_loss.npy"))
   # Eval_Average_PSNR = np.load(os.path.join(ResultSave_PATH,"Vaild_Average_PSNR.npy"))

    plot_chart(Train_PSNR,"Average PSNR in Training step","epochs","PSNR",save = False)
    plot_chart(Train_loss, "Average loss in Training step","epochs", "loss",save=False)
  #  plot_chart(Eval_Average_PSNR,"Average PSNR in evaluation step","epochs","PSNR",save=False)



