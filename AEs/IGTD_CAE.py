import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class IGTD_CAE(nn.Module):
    def __init__(self, n_features, n_classes, ksp_h, ksp_w):
        super(IGTD_CAE, self).__init__()

        class encoder(nn.Module):
            def __init__(self, n_features):
                super(encoder, self).__init__()
                self.start_img_size = (ksp_h['input'], ksp_w['input'])
                self.emb_dim = 50
                self.n_features = n_features
                self.criterion = nn.MSELoss()

                self.encoder_label_embedding = nn.Sequential(
                    nn.Embedding(n_classes, self.emb_dim),
                    nn.Linear(self.emb_dim, self.start_img_size[0] * self.start_img_size[1]),
                )

                self.lin_encoder = nn.Sequential(
                    nn.Linear(self.n_features, self.start_img_size[0] * self.start_img_size[1]),
                    nn.ReLU(True),
                )

                self.conv_encoder = nn.Sequential(
                    nn.ConvTranspose2d(2, 512, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"])),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.ConvTranspose2d(512, 1, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"])),
                    nn.Tanh(),
                )

                self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

            def forward(self, x, labels=None):
                labels = labels.long()
                label_embedding = self.encoder_label_embedding(labels).view(-1, 1, self.start_img_size[0], self.start_img_size[1])

                x = self.lin_encoder(x).view(-1, 1, self.start_img_size[0], self.start_img_size[1])

                return self.conv_encoder(torch.cat((x, label_embedding), dim=1))

            def train_model(self, dataloader, img_loader, epochs):
                self.train()
                total_loss = []
                for _ in tqdm(range(epochs), colour="yellow"):
                    for (features, f_labels), (img, _) in zip(dataloader, img_loader):
                        encoded_img = self.forward(features, f_labels)
                        self.optimizer.zero_grad()
                        loss = self.criterion(encoded_img, img)
                        loss.backward(retain_graph=True)
                        self.optimizer.step()
                        total_loss.append(loss.item())
                return total_loss

        class decoder(nn.Module):
            def __init__(self, n_features, ksp_h, ksp_w):
                super(decoder, self).__init__()
                self.start_img_size = (ksp_h['input'], ksp_w['input'])
                self.emb_dim = 50
                self.latent_img_size = (ksp_h["height"], ksp_w["width"])
                self.n_features = n_features
                self.criterion = nn.MSELoss()

                self.decoder_label_embedding = nn.Sequential(
                    nn.Embedding(n_classes, self.emb_dim),
                    nn.Linear(self.emb_dim, self.latent_img_size[0] * self.latent_img_size[1]),
                )

                self.conv_decoder = nn.Sequential(
                    nn.Conv2d(2, 1024, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"])),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(1024, 1024, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"])),
                    nn.BatchNorm2d(1024),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Flatten(),
                    nn.Dropout(0.2),

                    nn.Linear(1024 * self.start_img_size[0] * self.start_img_size[1], self.n_features),
                )

                self.optimizer = optim.Adam(self.parameters(), lr=0.00002)

            def forward(self, x, labels=None):
                labels = labels.long()
                label_embedding = self.decoder_label_embedding(labels).view(-1, 1, self.latent_img_size[0], self.latent_img_size[1])

                return self.conv_decoder(torch.cat((x, label_embedding), dim=1))

            def train_model(self, encoder, dataloader, img_loader, epochs):
                self.train()
                total_loss = []
                for _ in tqdm(range(epochs), colour="yellow"):
                    for (features, f_labels), (img, i_labels) in zip(dataloader, img_loader):
                        en_img = encoder(features, f_labels)
                        output_e = self.forward(en_img, f_labels)
                        loss_e = self.criterion(output_e, features)

                        output_r = self.forward(img, i_labels)
                        loss_r = self.criterion(output_r, features)

                        loss = loss_e + loss_r
                        self.optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        self.optimizer.step()
                        total_loss.append(loss.item())

                return total_loss

        self.encoder = encoder(n_features)
        self.decoder = decoder(n_features, ksp_h, ksp_w)

    def encode(self, x, labels):
        return self.encoder(x, labels)

    def decode(self, x, labels):
        return self.decoder(x, labels)

    def forward(self, x,  labels=None):
        return self.decode(self.encode(x, labels), labels)

    def train_model(self, dataloader, img_loader, epochs):
        encoder_loss = self.encoder.train_model(dataloader, img_loader, epochs)
        decoder_loss = self.decoder.train_model(self.encoder, dataloader, img_loader, epochs)

        return {"E": encoder_loss, "D": decoder_loss}
