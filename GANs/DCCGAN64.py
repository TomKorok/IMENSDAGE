# Deep Convolutional Conditional Generative Adversarial Network
import torch.nn as nn
import torch

class DCC_Generator64(nn.Module):
    def __init__(self, nz, n_classes):
        super(DCC_Generator64, self).__init__()
        self.emd_dim = 50
        self.start_img_size = 4
        self.channel_multiplier = 64

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emd_dim),
            nn.Linear(self.emd_dim, self.start_img_size * self.start_img_size),
        )

        self.lin_generator = nn.Sequential(
            nn.Linear(nz, self.channel_multiplier * 8 * self.start_img_size * self.start_img_size),
            nn.ReLU(True),
        )

        self.conv_generator = nn.Sequential(
            # state size. `(513) x 4 x 4`
            nn.ConvTranspose2d(self.channel_multiplier * 8 + 1, self.channel_multiplier * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 4),
            nn.ReLU(True),
            # state size. `(channel_multiplier * 4) x 8 x 8`
            nn.ConvTranspose2d(self.channel_multiplier * 4, self.channel_multiplier * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 2),
            nn.ReLU(True),
            # state size. `(channel_multiplier * 2) x 16 x 16`
            nn.ConvTranspose2d(self.channel_multiplier * 2, self.channel_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier),
            nn.ReLU(True),
            # state size. `(channel_multiplier) x 32 x 32`
            nn.ConvTranspose2d(self.channel_multiplier, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. `(3) x 64 x 64`
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.label_embedding(labels).view(-1, 1, self.start_img_size, self.start_img_size)

        x = self.lin_generator(x).view(-1, self.channel_multiplier * 8, self.start_img_size, self.start_img_size)
        x = torch.cat((x, label_embedding), dim=1)
        return self.conv_generator(x)


class DCC_Discriminator64(nn.Module):
    def __init__(self, n_classes):
        super(DCC_Discriminator64, self).__init__()
        self.emd_size = 50
        self.channel_multiplier = 64
        self.latent_img_size = 64

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emd_size),
            nn.Linear(self.emd_size, 1 * self.latent_img_size * self.latent_img_size),
        )

        self.conv_discriminator = nn.Sequential(
            # input is `(4) x 64 x 64`
            nn.Conv2d(4, self.channel_multiplier, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier) x 32 x 32`
            nn.Conv2d(self.channel_multiplier, self.channel_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier) x 16 x 16`
            nn.Conv2d(self.channel_multiplier, self.channel_multiplier * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier*2) x 8 x 8`
            nn.Conv2d(self.channel_multiplier * 2, self.channel_multiplier * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier*4) x 4 x 4`
            nn.Conv2d(self.channel_multiplier * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.label_embedding(labels).view(-1, 1, self.latent_img_size, self.latent_img_size)
        x = torch.cat((x, label_embedding), dim=1)
        x = self.conv_discriminator(x).squeeze(-1)
        return x.squeeze(-1)
