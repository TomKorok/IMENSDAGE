# Deep Convolutional Generative Adversarial Network
import torch.nn as nn

class DC_IGTD_Generator(nn.Module):
    def __init__(self, nz, ksp_h, ksp_w):
        super(DC_IGTD_Generator, self).__init__()
        self.ngf = 16
        self.conv_generator = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, self.ngf * 256, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf * 256),
            nn.ReLU(True),
            # state size. ``(ngf*8) x 4 x 4``
            nn.ConvTranspose2d(self.ngf * 256, self.ngf * 64, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.ngf * 64),
            nn.ReLU(True),
            # state size. ``(ngf*2) x 8 x 8`
            nn.ConvTranspose2d(self.ngf * 64, 1, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.Tanh()
            # state size. ``(nc) x 16 x 16``
        )

    def forward(self, x, label=None):
        x = x.unsqueeze(-1)
        return self.conv_generator(x.unsqueeze(-1))


class DC_IGTD_Discriminator(nn.Module):
    def __init__(self, ksp_h, ksp_w):
        super(DC_IGTD_Discriminator, self).__init__()
        self.ndf = 16
        self.main = nn.Sequential(
            # input is ``(nc) x 16 x 16``
            nn.Conv2d(1, self.ndf * 256, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(self.ndf * 256, self.ndf * 128, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.ndf * 128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(self.ndf * 128, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):
        x = self.main(x).squeeze(-1)
        return x.squeeze(-1)

