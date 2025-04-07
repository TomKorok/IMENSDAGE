# Deep Convolutional Generative Adversarial Network
import torch.nn as nn

class DCC_IGTD_Generator(nn.Module):
    def __init__(self, nz, ksp_h, ksp_w):
        super(DCC_IGTD_Generator, self).__init__()
        self.ngf = 16
      	emd_dim = 50
        self.start_img_s = 2

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, emd_dim),
            nn.Linear(emd_dim, self.start_img_s * self.start_img_s),
        )

        self.lin_generator = nn.Sequential(
            nn.Linear(nz, self.ngf * 256 * self.start_img_s * self.start_img_s),
            nn.ReLU(True),
        )
      
        self.conv_generator = nn.Sequential(
            # state size. ``(ngf*8) x 2 x 2``
            nn.ConvTranspose2d(self.ngf * 256, self.ngf * 64, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.ngf * 64),
            nn.ReLU(True),
            # state size. ``(ngf*2) x H1 x W2`
            nn.ConvTranspose2d(self.ngf * 64, 1, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.Tanh()
            # state size. ``(nc) x H2 x W2``
        )

    def forward(self, x, label=None):
        labels = labels.long()
        label_embedding = self.label_embedding(labels).view(-1, 1, self.start_img_s, self.start_img_s)

        x = self.lin_generator(x).view(-1, self.ngf * 256, self.start_img_s, self.start_img_s)
        x = torch.cat((x, label_embedding), dim=1)
        return self.conv_generator(x)


class DCC_IGTD_Discriminator(nn.Module):
    def __init__(self, ksp_h, ksp_w):
        super(DCC_IGTD_Discriminator, self).__init__()
        self.ndf = 16

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emd_size),
            nn.Linear(self.emd_size, 1 * self.ndf * self.ndf),
        )
      
        self.main = nn.Sequential(
            # input is ``(2) x H2 x W2``
            nn.Conv2d(2, self.ndf * 256, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x H1 x W1``
            nn.Conv2d(self.ndf * 256, self.ndf * 128, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.ndf * 128),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*8) x 2 x 2``
            nn.Conv2d(self.ndf * 128, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, label=None):
        labels = labels.long()
        label_embedding = self.label_embedding(labels).view(-1, 1, self.ndf, self.ndf)
        x = torch.cat((x, label_embedding), dim=1)
        x = self.conv_discriminator(x).squeeze(-1)
        return x.squeeze(-1)
      
