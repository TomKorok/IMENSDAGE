# Deep Convolutional Conditional Generative Adversarial Network

# Deep Convolutional Generative Adversarial Network
import torch.nn as nn
import torch

class DC_C_Generator(nn.Module):
    def __init__(self, nz, ngf, n_classes):
        super(DC_C_Generator, self).__init__()
        emd_dim = 10

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, emd_dim),
        )

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz+emd_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. ``(ngf*4) x 8 x 8``
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 16 x 16``
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. ``(ngf) x 32 x 32``
            nn.ConvTranspose2d( ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(nc) x 64 x 64``
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding(labels)
        x = torch.cat((x, label_embedding), dim=1)
        x = x.unsqueeze(-1)
        x = x.unsqueeze(-1)
        return self.main(x)


class DC_C_Discriminator(nn.Module):
    def __init__(self, nc, ndf, n_classes):
        super(DC_C_Discriminator, self).__init__()
        self.emd_size = 10
        self.ndf = ndf
        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emd_size),
        )

        self.discriminator1 = nn.Sequential(
            nn.Linear(self.emd_size, 1 * self.ndf * self.ndf),
        )

        self.discriminator2 = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(4, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding(labels)
        label_embedding = self.discriminator1(label_embedding).view(-1, 1, self.ndf, self.ndf)
        x = torch.cat([x, label_embedding], dim=1)
        x = self.discriminator2(x).squeeze(-1)
        return x.squeeze(-1)

