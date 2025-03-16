import pandas as pd
import os
from torch.utils.data import TensorDataset, DataLoader

from GANs import GANHandler
from AEs import AEHandler
from Support import DataHandler
from Support import HarryPlotter
import Evaluate


#IMage ENcoding for Synthetic DAta GEnaration

def filtering_samples(images, labels=None):
    df = pd.DataFrame({'index': range(len(labels)), 'label': labels.cpu().numpy()})

    # Select 5 samples per class
    selected_indices = df.groupby('label').head(5)['index'].values

    # Extract corresponding samples
    return images[selected_indices], labels[selected_indices].cpu().numpy()


class IMENSDAGE:
    def __init__(self, batch_size=128):
        self.device = HarryPlotter.print_default()
        self.batch_size = batch_size
        self.data_handler = None
        self.ae = None
        self.gen = None

        if not os.path.exists('results'):
            os.makedirs('results')

    def gen_gen_model(self, gan_model):
        gan = GANHandler.GANHandler(self.batch_size, gan_model["g_dim"], gan_model["d_dim"], self.data_handler.get_n_classes(), gan_model["model"]).to(self.device)
        gan.get_summary()
        return gan

    def gen_ae_model(self, ae_model):
        ae = AEHandler.AEHandler(ae_model["model"], ae_model["latent_size"], self.data_handler.get_n_features(), self.data_handler.get_n_classes()).to(self.device)
        ae.get_summary()
        return ae

    def read_data(self, location, round_exceptions, title, target=None):
        # Pipeline #1  reading data
        self.data_handler = DataHandler.DataHandler(self.batch_size, self.device, location, round_exceptions, title, target)

    def train_ae(self, ae_model=None):
        if ae_model is None:
            ae_model = {"model": "c", "latent_size": 64}
        self.ae = self.gen_ae_model(ae_model)
        total_ae_loss = self.ae.train_model(self.data_handler.get_dataloader(), epochs=1)
        HarryPlotter.plot_curve(f"Pre-trained AE Loss {self.data_handler.get_dataset_title()} AE={ae_model['model']}", total_ae_loss)
        return self.ae.encode(self.data_handler.get_feature_tensor(), self.data_handler.get_label_tensor()), self.data_handler.get_label_tensor()

    def train_gen_model(self, encoded_images, real_labels=None, gan_model=None):
        if gan_model is None:
            gan_model = {"model": "c", "g_dim": 28, "d_dim": 28}
        dataset = TensorDataset(encoded_images, real_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.gen = self.gen_gen_model(gan_model)
        total_d_loss, total_g_loss = self.gen.train_model(dataloader, epochs=1)
        HarryPlotter.plot_curve(f"Discriminator Loss {self.data_handler.get_dataset_title()}", total_d_loss)
        HarryPlotter.plot_curve(f"Generator Loss {self.data_handler.get_dataset_title()}", total_g_loss)
        return self.gen.sample(len(self.data_handler.get_dataframe())//self.data_handler.get_n_classes())

    def show_results(self, en_images, gen_images, en_labels=None, gen_labels=None):
        title = self.data_handler.get_dataset_title()
        n_classes = self.data_handler.get_n_classes()

        images, labels = filtering_samples(en_images, en_labels)
        HarryPlotter.plot_rgb_gigaplot(f"Encoded Images {title}", images, labels, n_classes)

        images, labels = filtering_samples(gen_images, gen_labels)
        HarryPlotter.plot_rgb_gigaplot(f"Generated Images {title}", images, labels, n_classes)

        HarryPlotter.display_text_samples(f"5 real samples per classes {title}", self.data_handler.get_real_samples(self.data_handler.get_n_classes()*5))
        HarryPlotter.display_text_samples(f"5 fake samples per classes {title}", self.data_handler.get_fake_samples(self.ae, gen_images, gen_labels, self.data_handler.get_n_classes()*5))

    def save_full_synth_set(self, gen_images, gen_labels=None):
        if not os.path.exists('results/synth_datasets'):
            os.makedirs('results/synth_datasets')
        dataframe = self.data_handler.get_fake_samples(self.ae, gen_images, gen_labels)
        dataframe.to_csv(f"results/synth_datasets/{self.data_handler.get_dataset_title()}_synth_dataset.csv", index=False)
        return dataframe

    def evaluate(self, synth_dataframes):
        Evaluate.evaluate(self.data_handler.get_location(), self.data_handler.is_classification(), synth_dataframes)
        return

    def fit(self, source, round_exceptions, title, target=None, ae_model=None, gan_model=None):
        # this function is a one-time callable function to run the entire framework
        # using one of the built-in GANs defined in the gan_model input
        # returns the synthetic dataset
        self.read_data(source, round_exceptions, title, target)
        en_images, en_labels = self.train_ae(ae_model=ae_model)
        gen_images, gen_labels = self.train_gen_model(en_images, en_labels, gan_model=gan_model)
        self.show_results(en_images, gen_images, en_labels=en_labels, gen_labels=gen_labels)
        return self.save_full_synth_set(gen_images, gen_labels)

