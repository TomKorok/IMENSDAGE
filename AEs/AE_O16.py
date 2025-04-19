import torch.nn as nn

class AE_O16(nn.Module):
    def __init__(self, n_features):
        super(AE_O16, self).__init__()
        self.start_img_size = 4

        self.lin_encoder = nn.Sequential(
            nn.Linear(n_features, self.start_img_size * self.start_img_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv_encoder = nn.Sequential(
            nn.ConvTranspose2d(1, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(3, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(512 * 4 * 4, n_features),
        )

    def encode(self, x, labels=None):
        x = self.lin_encoder(x).view(-1, 1, self.start_img_size, self.start_img_size)
        return self.conv_encoder(x)

    def decode(self, x, labels=None):
        return self.conv_decoder(x)

    def forward(self, x, labels=None):
        return self.decode(self.encode(x, labels), labels)

