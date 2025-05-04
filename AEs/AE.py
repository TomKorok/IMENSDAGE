import torch.nn as nn

class AE(nn.Module):
    def __init__(self, n_features):
        super(AE, self).__init__()
        self.channel_multiplier = 64

        self.conv_encoder = nn.Sequential(
            # input is features, going into a convolution
            nn.ConvTranspose2d(n_features, self.channel_multiplier * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*4) x 4 x 4``
            nn.ConvTranspose2d(self.channel_multiplier * 8, self.channel_multiplier * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 8 x 8``
            nn.ConvTranspose2d(self.channel_multiplier * 4, self.channel_multiplier * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 16 x 16``
            nn.ConvTranspose2d(self.channel_multiplier * 2, self.channel_multiplier, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent) x 32 x 32``
            nn.ConvTranspose2d(self.channel_multiplier, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(latent) x 64 x 64``
        )

        self.conv_decoder = nn.Sequential(
            # input is ``(latent) x 64 x 64``
            nn.Conv2d(3, self.channel_multiplier, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent) x 32 x 32``
            nn.Conv2d(self.channel_multiplier, self.channel_multiplier * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 16 x 16``
            nn.Conv2d(self.channel_multiplier * 2, self.channel_multiplier * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*2) x 8 x 8``
            nn.Conv2d(self.channel_multiplier * 4, self.channel_multiplier * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(latent*4) x 4 x 4``
            nn.Conv2d(self.channel_multiplier * 8, self.channel_multiplier * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.channel_multiplier * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(self.channel_multiplier * 16, n_features),
        )

    def encode(self, x, labels=None):
        x = x.unsqueeze(-1)
        return self.conv_encoder(x.unsqueeze(-1))

    def decode(self, x, labels=None):
        return self.conv_decoder(x)

    def forward(self, x, labels=None):
        return self.decode(self.encode(x, labels), labels)
