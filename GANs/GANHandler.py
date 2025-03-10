import torch.nn as nn
import torch
import torch.optim as optim
from tqdm.auto import tqdm
from torchinfo import summary

from GANs import CGAN
from GANs import GAN
from GANs import DCGAN
from GANs import DCCGAN

class GANHandler(nn.Module):
    def __init__(self, batch_size, g_dim, d_dim, n_classes, gan_model):
        super(GANHandler, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.noise_size = 100
        self.g_dim = g_dim
        self.d_dim = d_dim

        # init the networks and apply weight init
        if gan_model == 'dc':
            self.generator = DCGAN.DC_Generator(self.noise_size, g_dim)
            self.generator.apply(self.weights_init)
            self.discriminator = DCGAN.DC_Discriminator(d_dim)
            self.discriminator.apply(self.weights_init)
        else:
            self.generator = DCCGAN.DC_C_Generator(self.noise_size, g_dim, n_classes)
            self.generator.apply(self.weights_init)
            self.discriminator = DCCGAN.DC_C_Discriminator(self.noise_size, d_dim, n_classes)
            self.discriminator.apply(self.weights_init)

        self.criterion = nn.BCELoss() # the loss is BCELoss for both the D and G
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def get_labels(self, real=True):
        # real labels are 1 and fake labels 0
        if real:
            return torch.ones(self.batch_size, 1, device=self.device)
        else:
            return torch.zeros(self.batch_size, 1, device=self.device)

    def set_mode(self, mode):
        if mode == "train":
            self.generator.train()
            self.discriminator.train()
        elif mode == "eval":
            self.generator.eval()
            self.discriminator.eval()

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def get_generator(self):
        return self.generator

    def get_discriminator(self):
        return self.discriminator

    def sample(self, amount):
        self.set_mode("eval")
        class_labels = torch.arange(self.n_classes, device=self.device).repeat_interleave(amount)
        self.set_batch_size(class_labels.size(0))
        return self.generate(class_labels), class_labels

    def generate(self, labels):
        noise = torch.randn(self.batch_size, self.noise_size, device=self.device)
        return self.generator(noise, labels)

    def discriminate(self, x, labels):
        # adding a small noise to the labels
        return self.discriminator(x, labels + torch.rand(self.batch_size, device=self.device) * 0.00001)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def get_summary(self):
        print("Summary of the Generator")
        summary(self.generator, input_size=[(1, 100), (1,)])
        print("")
        print("Summary of the Discriminator")
        summary(self.discriminator, input_size=[(1, 3, self.d_dim, self.d_dim), (1,)])

    def opt_g(self, fake_result):
        g_loss = self.criterion(fake_result, self.get_labels(real=True))
        self.generator.zero_grad()
        g_loss.backward(retain_graph=True)
        self.g_optimizer.step()

        return g_loss.item()

    def opt_d(self, real_result, fake_result):
        loss_real = self.criterion(real_result, self.get_labels(real=True))
        loss_fake = self.criterion(fake_result, self.get_labels(real=False))
        d_loss = (loss_real + loss_fake) / 2
        self.discriminator.zero_grad()
        d_loss.backward(retain_graph=True)
        self.d_optimizer.step()

        return d_loss.item()

    def train_model(self, dataloader, epochs=1000):
        self.set_mode("train")
        total_g_loss = []
        total_d_loss = []
        for _ in tqdm(range(epochs), colour='magenta'):
            for real_images, real_labels in dataloader:
                batch_size = len(real_images)
                self.set_batch_size(batch_size)

                fake_labels = torch.randint(0, self.n_classes, (self.batch_size,), device=self.device)
                fake_images = self.generate(fake_labels)

                real_result = self.discriminate(real_images, real_labels)
                fake_result = self.discriminate(fake_images, fake_labels)

                # optimizing the discriminator
                total_d_loss.append(self.opt_d(real_result, fake_result))
                # optimizing the generator
                total_g_loss.append(self.opt_g(fake_result))

        return total_d_loss, total_g_loss
