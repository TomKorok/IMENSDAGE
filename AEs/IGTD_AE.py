import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class IGTD_AE(nn.Module):
    def __init__(self, n_features, n_classes):
        super(IGTD_AE, self).__init__()

