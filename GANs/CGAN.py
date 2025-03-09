# Conditional Generative Adversarial Network

import torch.nn as nn
import torch

class Conditional_Generator(nn.Module):
    def __init__(self, noise_size, n_classes, g_dim):
        super(Conditional_Generator, self).__init__()
        self.g_dim = g_dim
        self.out_channels = 3

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, 10),
        )

        self.generator1 = nn.Sequential(
            nn.Linear(noise_size+10, 512 * self.g_dim/4 * self.g_dim/4),
        )

        self.generator2 = nn.Sequential(
            nn.BatchNorm2d(512),

            nn.ConvTranspose2d(512, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(128, self.out_channels, 4, 2, 1, bias=False),
            nn.Tanh()

            # the output of the generator is an RGB image if input=7x7 => output=28x28 | if input=16x16 => output=64x64
        )

    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding(labels) # first label embedding
        x = torch.cat((x, label_embedding), dim=1) # adding the label to the noise
        x = self.generator1(x).view(-1, 512, self.g_dim/4, self.g_dim/4) # upscale to img * img * channels
        return self.generator2(x) # pass it through the convT layers


class Conditional_Discriminator(nn.Module):
    def __init__(self, d_dim, n_classes):
        super(Conditional_Discriminator, self).__init__()
        self.d_dim = d_dim
        self.in_channels = 3

        self.embedding = nn.Sequential(
            nn.Embedding(n_classes, 10),
        )

        self.discriminator1 = nn.Sequential(
            nn.Linear(10, 1 * self.d_dim * self.d_dim),
        )

        self.discriminator2 = nn.Sequential(
            nn.Conv2d(self.in_channels+1, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(256 * self.d_dim/4 * self.d_dim/4, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels):
        labels = labels.long()
        label_embedding = self.embedding(labels)
        label_embedding = self.discriminator1(label_embedding).view(-1, 1, self.d_dim, self.d_dim)
        x = torch.cat([x, label_embedding], dim=1)
        return self.discriminator2(x)