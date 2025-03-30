# Deep Convolutional Conditional Generative Adversarial Network

# Deep Convolutional Generative Adversarial Network
import torch.nn as nn
import torch

class DCC_Generator16(nn.Module):
    def __init__(self, nz, n_classes):
        super(DCC_Generator16, self).__init__()
        emd_dim = 50
        self.start_img_s = 4
        ngf = 16

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, emd_dim),
            nn.Linear(emd_dim, self.start_img_s * self.start_img_s),
        )

        self.lin_generator = nn.Sequential(
            nn.Linear(nz, 512 * self.start_img_s * self.start_img_s),
            nn.ReLU(True),
        )

        self.conv_generator = nn.Sequential(
            # state size. ``(ngf*8) x 4 x 4`
            nn.ConvTranspose2d(513, ngf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            # state size. ``(ngf) x 8 x 8`
            nn.ConvTranspose2d(ngf * 16, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(ngf*2) x 16 x 16`
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.label_embedding(labels).view(-1, 1, self.start_img_s, self.start_img_s)

        x = self.lin_generator(x).view(-1, 512, self.start_img_s, self.start_img_s)
        x = torch.cat((x, label_embedding), dim=1)
        return self.conv_generator(x)


class DCC_Discriminator16(nn.Module):
    def __init__(self, n_classes):
        super(DCC_Discriminator16, self).__init__()
        self.emd_size = 50
        self.ndf = 16

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emd_size),
            nn.Linear(self.emd_size, 1 * self.ndf * self.ndf),
        )

        self.conv_discriminator = nn.Sequential(
            # input is ``(nc) x 16 x 16``
            nn.Conv2d(4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(self.ndf * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.label_embedding(labels).view(-1, 1, self.ndf, self.ndf)
        x = torch.cat((x, label_embedding), dim=1)
        x = self.conv_discriminator(x).squeeze(-1)
        return x.squeeze(-1)
