import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

class IGTD_AE(nn.Module):
    def __init__(self, n_features, n_classes, ksp_h, ksp_w):
        super(IGTD_AE, self).__init__()
        self.n_classes = n_classes
        self.n_features = n_features

        class encoder(nn.Module):
            def __init__(self, n_features):
                super(encoder, self).__init__()
                self.start_img_size = (ksp_h['input'], ksp_w['input'])
                self.n_features = n_features
                self.criterion = nn.MSELoss()
                self.lin_encoder = nn.Sequential(
                    nn.Linear(self.n_features, self.start_img_size[0] * self.start_img_size[1]),
                    nn.LeakyReLU(0.2, inplace=True),
                )

                self.conv_encoder = nn.Sequential(
                    nn.ConvTranspose2d(1, 512, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"])),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.ConvTranspose2d(512, 1, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"])),
                    nn.Tanh(),
                )
                self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

            def forward(self, x, labels=None):
                x = self.lin_encoder(x).view(-1, 1, self.start_img_size, self.start_img_size)
                return self.conv_encoder(x)

            def train_model(self, dataloader, img_loader, epochs):
                self.train()
                total_loss = []
                for _ in tqdm(range(epochs), colour="yellow"):
                    for (features, _), (img, _) in zip(dataloader, img_loader):
                        encoded_img = self.forward(features)
                        self.optimizer.zero_grad()
                        loss = self.criterion(encoded_img, img)
                        loss.backward(retain_graph=True)
                        self.optimizer.step()
                        total_loss.append(loss.item())
                return total_loss

        class decoder(nn.Module):
            def __init__(self, n_features):
                super(decoder, self).__init__()
                self.start_img_size = (ksp_h['input'], ksp_w['input'])
                self.n_features = n_features
                self.criterion = nn.MSELoss()
                self.conv_decoder = nn.Sequential(
                    nn.Conv2d(1, 512, kernel_size=(ksp_h["k2"], ksp_w["k2"]), stride=(ksp_h["s2"], ksp_w["s2"]), padding=(ksp_h["p2"], ksp_w["p2"])),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Conv2d(512, 512, kernel_size=(ksp_h["k1"], ksp_w["k1"]), stride=(ksp_h["s1"], ksp_w["s1"]), padding=(ksp_h["p1"], ksp_w["p1"])),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2, inplace=True),

                    nn.Flatten(),
                    nn.Dropout(0.2),

                    nn.Linear(512 * self.start_img_size[0] * self.start_img_size[1], self.n_features),
                )
                self.optimizer = optim.Adam(self.parameters(), lr=0.00001)

            def forward(self, x, labels=None):
                return self.conv_decoder(x)

            def train_model(self, encoder, dataloader, img_loader, epochs):
                self.train()
                total_loss = []
                for _ in tqdm(range(epochs), colour="yellow"):
                    for (features, _), (img, _) in zip(dataloader, img_loader):
                        en_img = encoder(features)
                        output_e = self.forward(en_img)
                        loss_e = self.criterion(output_e, features)

                        output_r = self.forward(img)
                        loss_r = self.criterion(output_r, features)

                        loss = loss_e + loss_r
                        self.optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        self.optimizer.step()
                        total_loss.append(loss.item())

                return total_loss

        self.encoder = encoder(n_features)
        self.decoder = decoder(n_features)

    def encode(self, x, labels=None):
        return self.encoder(x, labels)

    def decode(self, x, labels=None):
        return self.decoder(x, labels)

    def forward(self, x, labels=None):
        return self.decode(self.encode(x, labels), labels)

    def train_model(self, dataloader, img_loader, epochs):
        encoder_loss = self.encoder.train_model(dataloader, img_loader, epochs)
        decoder_loss = self.decoder.train_model(self.encoder, dataloader, img_loader, epochs)

        return {"E": encoder_loss, "D": decoder_loss}
