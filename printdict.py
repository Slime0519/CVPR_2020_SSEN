import torch

if __name__ == "__main__":
    dict = torch.load("edsr_baseline_x2-1bc95232.pt")

    for name, param in dict.items():
        print(name)


        #print(param)
