# Deep Convolutional Conditional Generative Adversarial Network
import torch.nn as nn
import torch

class DCC_IGTD_Generator(nn.Module):
    def __init__(self, nz, n_classes, ksp_h, ksp_w):
        super(DCC_IGTD_Generator, self).__init__()
        self.channel_multiplier = 4
        self.emb_dim = 50
        self.start_img_size = (ksp_h['input'], ksp_w['input'])

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emb_dim),
            nn.Linear(self.emb_dim, self.start_img_size[0] * self.start_img_size[1]),
        )

        self.lin_generator = nn.Sequential(
            nn.Linear(nz, self.channel_multiplier * 128 * self.start_img_size[0] * self.start_img_size[1]),
            nn.ReLU(True),
        )
      
        self.conv_generator = nn.Sequential(
            # state size. `(channel_multiplier*256) x 2 x 2`
            nn.ConvTranspose2d(self.channel_multiplier * 128 + 1, self.channel_multiplier * 32, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 32),
            nn.ReLU(True),
            # state size. `(channel_multiplier*64) x H1 x W1`
            nn.ConvTranspose2d(self.channel_multiplier * 32, 1, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.Tanh()
            # state size. `(1) x H2 x W2`
        )

    def forward(self, x, label=None):
        label = label.long()
        label_embedding = self.label_embedding(label).view(-1, 1, self.start_img_size[0], self.start_img_size[1])

        x = self.lin_generator(x).view(-1, self.channel_multiplier * 128, self.start_img_size[0], self.start_img_size[1])
        x = torch.cat((x, label_embedding), dim=1)
        return self.conv_generator(x)


class DCC_IGTD_Discriminator(nn.Module):
    def __init__(self, n_classes, ksp_h, ksp_w):
        super(DCC_IGTD_Discriminator, self).__init__()
        self.channel_multiplier = 1
        self.emb_dim = 50
        self.latent_img_size = (ksp_h['height'], ksp_w['width'])

        self.label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emb_dim),
            nn.Linear(self.emb_dim, 1 * self.latent_img_size[0] * self.latent_img_size[1]),
        )
      
        self.conv_discriminator = nn.Sequential(
            # input is `(2) x H2 x W2`
            nn.Conv2d(2, self.channel_multiplier * 32, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"]), bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier*256) x H1 x W1`
            nn.Conv2d(self.channel_multiplier * 32, self.channel_multiplier * 8, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"]), bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. `(channel_multiplier*128) x 2 x 2`
            nn.Conv2d(self.channel_multiplier * 8, 1, kernel_size=(ksp_h["input"], ksp_w["input"]), stride=1, padding=0, bias=False),
            nn.Sigmoid()
            # state size. `1`
        )

    def forward(self, x, labels=None):
        labels = labels.long()
        label_embedding = self.label_embedding(labels).view(-1, 1, self.latent_img_size[0], self.latent_img_size[1])
        x = torch.cat((x, label_embedding), dim=1)
        x = self.conv_discriminator(x).squeeze(-1)
        return x.squeeze(-1)
      
