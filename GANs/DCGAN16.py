# Deep Convolutional Generative Adversarial Network
import torch.nn as nn

class DC_Generator16(nn.Module):
    def __init__(self, nz):
        super(DC_Generator16, self).__init__()
        self.channel_multiplier = 16
        self.conv_generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, self.channel_multiplier * 32, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 32),
            nn.ReLU(True),
            # state size. `(channel_multiplier*8) x 4 x 4`
            nn.ConvTranspose2d(self.channel_multiplier * 32, self.channel_multiplier * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 8),
            nn.ReLU(True),
            # state size. `(channel_multiplier*2) x 8 x 8`
            nn.ConvTranspose2d(self.channel_multiplier * 8, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. `(3) x 16 x 16`
        )

    def forward(self, x, label=None):
        x = x.unsqueeze(-1)
        return self.conv_generator(x.unsqueeze(-1))


class DC_Discriminator16(nn.Module):
    def __init__(self):
        super(DC_Discriminator16, self).__init__()
        self.channel_multiplier = 16
        self.main = nn.Sequential(
            # input is `(3) x 16 x 16`
            nn.Conv2d(3, self.channel_multiplier * 4, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier * 8) x 8 x 8`
            nn.Conv2d(self.channel_multiplier * 4, self.channel_multiplier * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier * 16) x 4 x 4`
            nn.Conv2d(self.channel_multiplier * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):
        x = self.main(x).squeeze(-1)
        return x.squeeze(-1)

