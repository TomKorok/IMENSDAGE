import torch.nn as nn
from torchinfo import summary
import torch.optim as optim
from tqdm.auto import tqdm

from AEs import AE, AE_O16, AE_O64, CAE, CAE_O16, CAE_O64, IGTD_AE, IGTD_CAE


def create_ae_model(n_features, n_classes, ae_model, ksp_h=None, ksp_w=None):
    match ae_model:
        case 'c':
            return CAE.CAE(n_features, n_classes), False
        case 'n':
            return AE.AE(n_features), False
        case 'n_igtd':
            return IGTD_AE.IGTD_AE(n_features, n_classes, ksp_h, ksp_w), True
        case 'c_igtd':
            return IGTD_CAE.IGTD_CAE(n_features, n_classes, ksp_h, ksp_w), True
        case 'co64':
            return CAE_O64.CAE_O64(n_features, n_classes), False
        case 'no64':
            return AE_O64.AE_O64(n_features), False
        case 'co16':
            return CAE_O16.CAE_O16(n_features, n_classes), False
        case 'no16':
            return AE_O16.AE_O16(n_features), False
        case _:
            return CAE.CAE(n_features, n_classes), False

class AEHandler(nn.Module):
    def __init__(self, ae_model, n_features, n_classes, ksp_h=None, ksp_w=None):
        super(AEHandler, self).__init__()
        self.autoencoder, self.igtd = create_ae_model(n_features, n_classes, ae_model, ksp_h, ksp_w)
        self.n_features = n_features
        self.n_classes = n_classes
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.00001)

    def encode(self, x, labels):
        return self.autoencoder.encode(x, labels)

    def decode(self, x, labels):
        return self.autoencoder.decode(x, labels)

    def train_model(self, dataloader, img_loader=None, epochs=50):
        if self.igtd:
            return self.autoencoder.train_model(dataloader, img_loader, epochs)
        else:
            self.autoencoder.train()
            total_loss = []
            for _ in tqdm(range(epochs), colour="yellow"):
                epoch_loss = 0
                for features, labels in dataloader:
                    output = self.autoencoder.forward(features, labels)
                    self.optimizer.zero_grad()
                    loss = self.criterion(output, features)
                    loss.backward(retain_graph=True)
                    self.optimizer.step()
                    epoch_loss += loss.item()
                total_loss.append(epoch_loss)

            return {"AE": total_loss}

    def get_summary(self):
        print("Summary of the Autoencoder")
        summary(self.autoencoder, input_size=[(1, self.n_features), (1,)])
