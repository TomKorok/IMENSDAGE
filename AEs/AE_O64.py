import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class AE_O64(nn.Module):
    def __init__(self, n_features, n_classes):
        super(AE_O64, self).__init__()
        self.start_img_s = 4
        self.latent_img_s = 64
        self.n_classes = n_classes
        self.n_features = n_features
        self.criterion = nn.MSELoss()

        self.lin_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.start_img_s * self.start_img_s),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(256, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(256 * 4 * 4, self.n_features),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def encode(self, x, labels=None):
        x = self.lin_encoder(x).view(-1, 1, self.start_img_s, self.start_img_s)
        return self.conv_encoder(x)

    def decode(self, x, labels=None):
        return self.conv_decoder(x)

    def forward(self, x, labels=None):
        return self.decode(self.encode(x, labels), labels)

    def train_model(self, dataloader, img_loader, epochs):
        self.train()
        total_loss = []
        for _ in tqdm(range(epochs), colour="yellow"):
            for features, _ in dataloader:
                output = self.forward(features)
                self.optimizer.zero_grad()
                loss = self.criterion(output, features)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss.append(loss.item())

        return {"AE":total_loss}
