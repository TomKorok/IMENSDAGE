import numpy as np
import torch
from matplotlib import pyplot as plt
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import csv

def plot_rgb_gigaplot(title, images, en_labels, n_classes):
    examples_per_class = int(images.shape[0] / n_classes)
    # Scale from [-1, 1] to [0, 255]
    images = ((images + 1) * 127.5).clip(0, 255).to(torch.uint8)

    fig, axes = plt.subplots(n_classes, examples_per_class, figsize=(examples_per_class, n_classes))
    if n_classes == 1:
        axes = np.expand_dims(axes, axis=0)
    for i in range(n_classes):
        for j in range(examples_per_class):
            idx = i * examples_per_class + j
            img = images[idx].detach().cpu().permute(1, 2, 0).numpy()  # Convert to numpy and rearrange channels
            axes[i, j].imshow(img)  # No cmap, since it's color
            axes[i, j].axis('off')
            axes[i, j].set_title(en_labels[idx], fontsize=8, pad=5)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(hspace=0.5)
    plt.suptitle(title, fontsize=12, y=1)
    plt.savefig(f'results/synth_img/{title}.png')
    plt.show()

def plot_curve(title, curve_dict):
    for key in curve_dict:
        curve = np.convolve(curve_dict[key], np.ones((1,)) / 1, mode='valid')
        plt.plot([j for j in range(len(curve))], curve, alpha=1, label=key)
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(f"results/curves/{title}.png")
    plt.show()

def print_default():
    # getting and setting and displaying the used device
    try:
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        print(f"Name of current CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        print(e)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def display_text_samples(title, dataframe):
    print("")
    print(f"{title}:")
    print(dataframe.head(len(dataframe)).to_string())

def calculate_fid(act1, act2, filename):
    act1 = act1.view(act1.size(0), -1)
    act2 = act2.view(act2.size(0), -1)
    # down sampling for faster computation
    if act1.shape[1] > 1000:
        return "image size was too big so FID score is not calculated"
    else:
        mu1 = torch.mean(act1, dim=0).detach().cpu().numpy()
        mu2 = torch.mean(act2, dim=0).detach().cpu().numpy()
        # calculate mean and covariance statistics
        sigma1 = cov(act1.detach().cpu().numpy(), rowvar=False)
        sigma2 = cov(act2.detach().cpu().numpy(), rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        with open(f"results/metrics/{filename}.csv", mode='w', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerow(['FID Score'])  # header
            writer.writerow([f"{fid:.4f}".replace('.', ',')])
        return fid

