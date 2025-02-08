import torch
import os

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

    def read_data(self, location, title, target=None, classification=True):
        # Pipeline #1  reading data
        self.data_handler = DataHandler.DataHandler(self.batch_size, self.device, location, title, target, classification)
        if classification:
            self.n_classes = self.data_handler.get_n_classes()

    def train_ae(self, ae_model='c'):
        self.ae = self.gen_ae_model(ae_model)
        total_ae_loss = self.ae.train_model(self.data_handler.get_dataloader(), epochs=1)
        HarryPlotter.plot_curve(f"Pre-trained AE Loss {self.data_handler.get_dataset_title()} AE={ae_model}", total_ae_loss)
        return self.ae.encode(self.data_handler.get_features(), self.data_handler.get_labels())

    def train_gen_model(self, encoded_images):
        self.gen = self.gen_gen_model()
        total_d_loss, total_g_loss = self.gen.train_model(encoded_images, self.data_handler.get_dataloader(), epochs=1)
        HarryPlotter.plot_curve(f"Discriminator Loss {self.data_handler.get_dataset_title()}", total_d_loss)
        HarryPlotter.plot_curve(f"Generator Loss {self.data_handler.get_dataset_title()}", total_g_loss)
        return self.gen.sample(len(self.data_handler.get_dataframe())//2)

    def show_results(self, encoded_images, gen_data, round_exceptions): #TODO random 10 samples
        HarryPlotter.plot_rgb_gigaplot(f"Encoded Images {self.data_handler.get_dataset_title}", encoded_images, self.n_classes)
        HarryPlotter.plot_rgb_gigaplot(f"Generated Images {self.data_handler.get_dataset_title}", gen_data["gen_images"], self.n_classes) #TODO

        HarryPlotter.display_text_samples(f"10 Real Samples {self.data_handler.get_dataset_title}", self.data_handler.get_real_samples())
        HarryPlotter.display_text_samples(f"10 Fake Samples {self.data_handler.get_dataset_title}", self.data_handler.get_fake_samples(self.ae, gen_data, round_exceptions))


    def evaluate(self):
        pass