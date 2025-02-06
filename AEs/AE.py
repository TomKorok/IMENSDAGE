import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class AE(nn.Module):
    def __init__(self, n_features, n_classes):
        super(AE, self).__init__()
        self.start_img_s = 7
        self.latent_img_s = 28
        self.n_classes = n_classes
        self.n_features = n_features
        self.criterion = nn.MSELoss()

        self.lin_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.start_img_s * self.start_img_s),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
        )

        self.conv_encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 256, kernel_size=4, stride=2, padding=1),
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

            nn.Flatten(),

            nn.Linear(256 * 7 * 7, self.n_features),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def encode(self, x, labels=None):
        if x.size(1) != self.n_features:
            labels = labels.unsqueeze(1)
            x = torch.cat((x, labels), dim=1)
        x = self.lin_encoder(x)
        x = x.view(-1, 1, self.start_img_s, self.start_img_s)
        return self.conv_encoder(x)

    def decode(self, x, labels=None):
        return self.conv_decoder(x)

    def forward(self, x, labels):
        return self.decode(self.encode(x, labels), labels)

    def train_model(self, dataloader, epochs):
        self.train()
        total_loss = []
        for _ in tqdm(range(epochs), colour="yellow"):
            for features, labels in dataloader:
                output = self.forward(features, labels)
                self.optimizer.zero_grad()
                loss = self.criterion(output, features)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss.append(loss.item())

        return total_loss

    def get_n_features(self):
        return self.n_features