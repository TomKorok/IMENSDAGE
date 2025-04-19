# Deep Convolutional Generative Adversarial Network
import torch.nn as nn

class DC_IGTD_Generator(nn.Module):
    def __init__(self, nz, ksp_h, ksp_w):
        super(DC_IGTD_Generator, self).__init__()
        self.channel_multiplier = 4
        self.conv_generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, self.channel_multiplier * 128, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 128),
            nn.ReLU(True),
            # state size. `(channel_multiplier * 256) x 2 x 2`
            nn.ConvTranspose2d(self.channel_multiplier * 128, self.channel_multiplier * 32, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 32),
            nn.ReLU(True),
            # state size. `(channel_multiplier * 64) x H1 x W1`
            nn.ConvTranspose2d(self.channel_multiplier * 32, 1, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.Tanh()
            # state size. `(1) x H2 x W2`
        )

    def forward(self, x, label=None):
        x = x.unsqueeze(-1)
        return self.conv_generator(x.unsqueeze(-1))


class DC_IGTD_Discriminator(nn.Module):
    def __init__(self, ksp_h, ksp_w):
        super(DC_IGTD_Discriminator, self).__init__()
        self.channel_multiplier = 4
        self.main = nn.Sequential(
            # input is `(1) x H2 x W2`
            nn.Conv2d(1, self.channel_multiplier * 16, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier * 256) x H1 x W1`
            nn.Conv2d(self.channel_multiplier * 16, self.channel_multiplier * 64, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 64),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier * 128) x 2 x 2`
            nn.Conv2d(self.channel_multiplier * 64, 1, kernel_size=(ksp_h["input"], ksp_w["input"]), stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):
        x = self.main(x).squeeze(-1)
        return x.squeeze(-1)

