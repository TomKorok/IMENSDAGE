# Generative Adversarial Network

import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size, g_mid_img_s):
        super(Generator, self).__init__()
        self.g_mid_img_s = g_mid_img_s
        self.out_channels = 3

        self.generator1 = nn.Sequential(
            nn.Linear(noise_size, 512 * self.g_mid_img_s * self.g_mid_img_s),
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

    def forward(self, x, labels=None):
        x = self.generator1(x).view(-1, 512, self.g_mid_img_s, self.g_mid_img_s) # upscale to img * img * channels
        return self.generator2(x) # pass it through the convT layers

class Discriminator(nn.Module):
    def __init__(self, d_dim):
        super(Discriminator, self).__init__()
        self.d_dim = d_dim
        self.in_channels = 3

        self.discriminator = nn.Sequential(
            nn.Conv2d(self.in_channels, 128, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.4),

            nn.Linear(256 * self.d_dim/4 * self.d_dim/4, 1),
            nn.Sigmoid()
        )
    def forward(self, x, labels=None):
        return self.discriminator(x)
