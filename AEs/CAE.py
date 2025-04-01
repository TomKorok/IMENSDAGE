import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class CAE(nn.Module):
    def __init__(self, n_features, n_classes):
        super(CAE, self).__init__()
        self.latent_size = 64
        self.n_classes = n_classes
        self.n_features = n_features
        self.criterion = nn.MSELoss()
        self.emb_size = 50
        self.start_img_s = 4

        #encoder network
        self.encoder_label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emb_size),
            nn.Linear(self.emb_size, self.start_img_s * self.start_img_s),
        )

        self.lin_encoder = nn.Sequential(
            nn.Linear(self.n_features, 512 * self.start_img_s * self.start_img_s),
            nn.ReLU(True),
        )

        self.conv_encoder = nn.Sequential(
            # state size. ``(latent*4) x 4 x 4``
            nn.ConvTranspose2d(513, self.latent_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 4),
            nn.ReLU(True),
            # state size. ``(latent*2) x 8 x 8``
            nn.ConvTranspose2d(self.latent_size * 4, self.latent_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 2),
            nn.ReLU(True),
            # state size. ``(latent*2) x 16 x 16``
            nn.ConvTranspose2d(self.latent_size * 2, self.latent_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU(True),
            # state size. ``(latent) x 32 x 32``
            nn.ConvTranspose2d(self.latent_size, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. ``(latent) x 64 x 64``
        )

        # Decoder network
        self.decoder_label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emb_size),
            nn.Linear(self.emb_size, self.latent_size * self.latent_size),
        )

        self.conv_decoder = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(4, self.latent_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(self.latent_size, self.latent_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(self.latent_size * 2, self.latent_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*2) x 8 x 8``
            nn.Conv2d(self.latent_size * 4, self.latent_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.latent_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. ``(ndf*4) x 4 x 4``
            nn.Conv2d(self.latent_size * 8, self.latent_size * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(self.latent_size * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Dropout(0.2),

            nn.Linear(self.latent_size * 16, self.n_features),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

    def encode(self, x, labels):
        labels = labels.long()
        label_embedding = self.encoder_label_embedding(labels).view(-1, 1, self.start_img_s, self.start_img_s)

        x = self.lin_encoder(x).view(-1, 512, self.start_img_s, self.start_img_s)
        return self.conv_encoder(torch.cat((x, label_embedding), dim=1))

    def decode(self, x, labels):
        labels = labels.long()
        label_embedding = self.decoder_label_embedding(labels).view(-1, 1, self.latent_size, self.latent_size)

        return self.conv_decoder(torch.cat((x, label_embedding), dim=1))

    def forward(self, x,  labels=None):
        return self.decode(self.encode(x, labels), labels)

    def train_model(self, dataloader, img_loader, epochs):
        self.train()
        total_loss = []
        for _ in tqdm(range(epochs), colour="yellow"):
            for features, labels in dataloader:
                output = self.forward(features, labels)
                self.optimizer.zero_grad()
                loss = self.criterion(output, features)
                loss.backward(retain_graph=True)
                self.optimizer.step()
                total_loss.append(loss.item())

        return {"AE":total_loss}
