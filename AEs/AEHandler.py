import torch.nn as nn
from torchinfo import summary

from AEs import AE, AE_O16, AE_O64, CAE, CAE_O16, CAE_O64, IGTD_AE, IGTD_CAE


def create_ae_model(n_features, n_classes, ae_model, ksp_h=None, ksp_w=None):
    match ae_model:
        case 'c':
            return CAE.CAE(n_features, n_classes)
        case 'n':
            return AE.AE(n_features, n_classes)
        case 'n_igtd':
            return IGTD_AE.IGTD_AE(n_features, n_classes, ksp_h, ksp_w)
        case 'c_igtd':
            return IGTD_CAE.IGTD_CAE(n_features, n_classes, ksp_h, ksp_w)
        case 'co64':
            return CAE_O64.CAE_O64(n_features, n_classes)
        case 'no64':
            return AE_O64.AE_O64(n_features, n_classes)
        case 'co16':
            return CAE_O16.CAE_O16(n_features, n_classes)
        case 'no16':
            return AE_O16.AE_O16(n_features, n_classes)
        case _:
            return CAE.CAE(n_features, n_classes)

class AEHandler(nn.Module):
    def __init__(self, ae_model, n_features, n_classes, ksp_h=None, ksp_w=None):
        super(AEHandler, self).__init__()
        self.autoencoder = create_ae_model(n_features, n_classes, ae_model, ksp_h, ksp_w)
        self.n_features = n_features
        self.n_classes = n_classes

    def encode(self, x, labels):
        return self.autoencoder.encode(x, labels)

    def decode(self, x, labels):
        return self.autoencoder.decode(x, labels)

    def train_model(self, dataloader, img_loader=None, epochs=50):
        return self.autoencoder.train_model(dataloader, img_loader, epochs)

    def get_summary(self):
        print("Summary of the Autoencoder")
        summary(self.autoencoder, input_size=[(1, self.n_features), (1,)])
