import torch.nn as nn
from torchinfo import summary

from AEs import CAE
from AEs import AE
from AEs import IGTD_AE


def create_ae_model(n_features, n_classes, conditional):
    match conditional:
        case 'c':
            return CAE.CAE(n_features, n_classes)
        case 'n':
            return AE.AE(n_features, n_classes)
        case 'igtd':
            return IGTD_AE.IGTD_AE(n_features, n_classes)
        case _:
            return CAE.CAE(n_features, n_classes)

class AEHandler(nn.Module):
    def __init__(self, conditional, n_features, n_classes):
        super(AEHandler, self).__init__()
        self.conditional = conditional
        self.autoencoder = create_ae_model(n_features, n_classes, conditional)
        self.n_features = self.autoencoder.get_n_features()
        self.n_classes = n_classes


    def encode(self, x, labels):
        return self.autoencoder.encode(x, labels)

    def decode(self, x, labels):
        return self.autoencoder.decode(x, labels)

    def train_model(self, dataloader, epochs=50):
        return self.autoencoder.train_model(dataloader, epochs)

    def get_summary(self):
        print("Summary of the Autoencoder")
        summary(self.autoencoder, input_size=[(1, self.n_features), (1,)])

    def get_conditional(self):
        return self.conditional



