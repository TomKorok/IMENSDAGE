import torch
import os

from torch.utils.data import TensorDataset, DataLoader

import GAN
from AEs import AEHandler
from Support import DataHandler
from Support import HarryPlotter


#IMage ENcoding for Synthetic DAta GEnaration

class IMENSDAGE:
    def __init__(self, batch_size=128):
        self.device = HarryPlotter.print_default()
        self.n_classes = None
        self.batch_size = batch_size
        self.data_handler = None
        self.ae = None
        self.gen = None

        if not os.path.exists('results'):
            os.makedirs('results')

    def gen_gen_model(self):
        gan = GAN.GAN(self.batch_size, 28, 7, 7, 2).to(self.device)
        gan.get_summary()
        return gan

    def gen_ae_model(self, ae_model):
        ae = AEHandler.AEHandler(ae_model, self.data_handler.get_n_features(), self.n_classes).to(self.device)
        ae.get_summary()
        return ae

    def read_data(self, location, round_exceptions, title, target=None, classification=True):
        # Pipeline #1  reading data
        self.data_handler = DataHandler.DataHandler(self.batch_size, self.device, location, round_exceptions, title, target, classification)
        if classification:
            self.n_classes = self.data_handler.get_n_classes()

    def train_ae(self, ae_model='c'):
        self.ae = self.gen_ae_model(ae_model)
        total_ae_loss = self.ae.train_model(self.data_handler.get_dataloader(), epochs=1)
        HarryPlotter.plot_curve(f"Pre-trained AE Loss {self.data_handler.get_dataset_title()} AE={ae_model}", total_ae_loss)
        return {"encoded_images": self.ae.encode(self.data_handler.get_feature_tensor(), self.data_handler.get_label_tensor()), "real_labels": self.data_handler.get_label_tensor()}

    def train_gen_model(self, encoded_images, real_labels):
        if real_labels is None:
            print("This is a CGAN model which requires real labels")
            return
        else:
            dataset = TensorDataset(encoded_images, real_labels)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            self.gen = self.gen_gen_model()
            total_d_loss, total_g_loss = self.gen.train_model(dataloader, epochs=1)
            HarryPlotter.plot_curve(f"Discriminator Loss {self.data_handler.get_dataset_title()}", total_d_loss)
            HarryPlotter.plot_curve(f"Generator Loss {self.data_handler.get_dataset_title()}", total_g_loss)
            return self.gen.sample(len(self.data_handler.get_dataframe())//2)

    def show_results(self, encoded_images, gen_data):
        ten_real_images = torch.cat((encoded_images[:5], encoded_images[encoded_images.shape[0] // 2:encoded_images.shape[0] // 2 + 5]))
        HarryPlotter.plot_rgb_gigaplot(f"Encoded Images {self.data_handler.get_dataset_title()}", ten_real_images, self.n_classes)

        ten_fake_images = torch.cat((gen_data["fake_images"][:5], gen_data["fake_images"][gen_data["fake_images"].shape[0] // 2:gen_data["fake_images"].shape[0] // 2 + 5]))
        HarryPlotter.plot_rgb_gigaplot(f"Generated Images {self.data_handler.get_dataset_title()}", ten_fake_images, self.n_classes)

        HarryPlotter.display_text_samples(f"10 Real Samples {self.data_handler.get_dataset_title()}", self.data_handler.get_real_samples(5))
        HarryPlotter.display_text_samples(f"10 Fake Samples {self.data_handler.get_dataset_title()}", self.data_handler.get_fake_samples(self.ae, gen_data, 5))

    def get_full_fake_set(self, gen_data):
        self.data_handler.get_fake_samples(self.ae, gen_data)

    def evaluate(self):
        pass