import torch
import torch.nn as nn


class CAE_O16(nn.Module):
    def __init__(self, n_features, n_classes):
        super(CAE_O16, self).__init__()
        self.start_img_size = 4
        self.latent_img_size = 16
        self.emb_dim = 50

        #encoder network
        self.encoder_label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emb_dim),
            nn.Linear(self.emb_dim, self.start_img_size * self.start_img_size),
        )
        self.lin_encoder = nn.Sequential(
            nn.Linear(n_features, self.start_img_size * self.start_img_size),
            nn.ReLU(True),
        )

        self.conv_encoder = nn.Sequential(
            nn.ConvTranspose2d(2, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(512, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

        # Decoder network
        self.decoder_label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emb_dim),
            nn.Linear(self.emb_dim, self.latent_img_size * self.latent_img_size),
        )

        self.conv_decoder = nn.Sequential(
            nn.Conv2d(4, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(512 * 4 * 4, n_features),
        )

    def encode(self, x, labels):
        labels = labels.long()
        label_embedding = self.encoder_label_embedding(labels).view(-1, 1, self.start_img_size, self.start_img_size)

        x = self.lin_encoder(x).view(-1, 1, self.start_img_size, self.start_img_size)

        return self.conv_encoder(torch.cat((x, label_embedding), dim=1))

    def decode(self, x, labels):
        labels = labels.long()
        label_embedding = self.decoder_label_embedding(labels).view(-1, 1, self.latent_img_size, self.latent_img_size)

        return self.conv_decoder(torch.cat((x, label_embedding), dim=1))

    def forward(self, x, labels):
        return self.decode(self.encode(x, labels), labels)
