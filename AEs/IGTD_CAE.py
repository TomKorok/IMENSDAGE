import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class IGTD_CAE(nn.Module):
    def __init__(self, n_features, n_classes, ksp_h, ksp_w):
        super(IGTD_CAE, self).__init__()
        self.start_img_s = 4
        self.latent_img_s = 16
        self.n_classes = n_classes
        self.n_features = n_features
        self.criterion = nn.MSELoss()
        self.emb_dim = 50

        #encoder network
        self.encoder_label_embedding = nn.Sequential(
            nn.Embedding(n_classes, self.emb_dim),
            nn.Linear(self.emb_dim, self.start_img_s * self.start_img_s),
        )
        self.lin_encoder = nn.Sequential(
            nn.Linear(self.n_features, self.start_img_s * self.start_img_s),
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
            nn.Linear(self.emb_dim, self.latent_img_s * self.latent_img_s),
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

            nn.Linear(512 * 4 * 4, self.n_features),
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.00002)

    def encode(self, x, labels):
        labels = labels.long()
        label_embedding = self.encoder_label_embedding(labels).view(-1, 1, self.start_img_s, self.start_img_s)

        x = self.lin_encoder(x).view(-1, 1, self.start_img_s, self.start_img_s)

        return self.conv_encoder(torch.cat((x, label_embedding), dim=1))

    def decode(self, x, labels):
        labels = labels.long()
        label_embedding = self.decoder_label_embedding(labels).view(-1, 1, self.latent_img_s, self.latent_img_s)

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

        return total_loss
