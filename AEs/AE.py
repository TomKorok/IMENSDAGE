import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class AE(nn.Module):
    def __init__(self, n_features, n_classes):
        super(AE, self).__init__()
        self.latent_size = 64
        self.n_classes = n_classes
        self.n_features = n_features
        self.criterion = nn.MSELoss()

        self.conv_encoder = nn.Sequential(
            # input is features, going into a convolution
            nn.ConvTranspose2d(self.n_features, self.latent_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.latent_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*4) x 4 x 4``
            nn.ConvTranspose2d(self.latent_size * 8, self.latent_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 8 x 8``
            nn.ConvTranspose2d(self.latent_size * 4, self.latent_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 16 x 16``
            nn.ConvTranspose2d(self.latent_size * 2, self.latent_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent) x 32 x 32``
            nn.ConvTranspose2d(self.latent_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(latent) x 64 x 64``
        )

        self.conv_decoder = nn.Sequential(
            # input is ``(latent) x 64 x 64``
            nn.Conv2d(3, self.latent_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent) x 32 x 32``
            nn.Conv2d(self.latent_size, self.latent_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 16 x 16``
            nn.Conv2d(self.latent_size * 2, self.latent_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 8 x 8``
            nn.Conv2d(self.latent_size * 4, self.latent_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*4) x 4 x 4``
            nn.Conv2d(self.latent_size * 8, self.latent_size * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.latent_size * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),

            nn.Linear(self.latent_size * 16, self.n_features),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def encode(self, x, labels=None):
        x = x.unsqueeze(-1)
        return self.conv_encoder(x.unsqueeze(-1))

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
