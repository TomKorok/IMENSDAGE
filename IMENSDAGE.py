import pandas as pd
import os

import torch
from torch.utils.data import TensorDataset, DataLoader
import math

from GANs import GANHandler
from AEs import AEHandler
from Support import DataHandler, HarryPlotter, KSP_LookUp


#IMage ENcoding for Synthetic DAta GEnaration

def filtering_samples(images, labels=None):
    df = pd.DataFrame({'index': range(len(labels)), 'label': labels.cpu().numpy()})

    # Select 5 samples per class
    selected_indices = df.groupby('label').head(5)['index'].values

    # Extract corresponding samples
    return images[selected_indices], labels[selected_indices].cpu().numpy()


def closest_factors(n):
    sqrt_n = int(math.sqrt(n))

    # Start from the square root and move down to find the closest factor pair
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return  n // i, i  # n//i = height & i = width
    return n, 1


class IMENSDAGE:
    def __init__(self, batch_size=32):
        self.device = HarryPlotter.print_default()
        self.batch_size = batch_size
        self.data_handler = None
        self.ae_model = None
        self.ae = None
        self.gen_model = None
        self.gen = None
        self.title = None
        self.ksp_h, self.ksp_w = None, None

        # set the output folders
        if not os.path.exists('results'):
            os.makedirs('results')
        if not os.path.exists('results/synth_img'):
            os.makedirs('results/synth_img')
        if not os.path.exists('results/curves'):
            os.makedirs('results/curves')
        if not os.path.exists('results/synth_datasets'):
            os.makedirs('results/synth_datasets')
        if not os.path.exists('results/metrics'):
            os.makedirs('results/metrics')
        if not os.path.exists('results/stats'):
            os.makedirs('results/stats')

    def gen_gen_model(self, gan_model):
        gan = GANHandler.GANHandler(self.batch_size, self.data_handler.get_n_classes(), gan_model, self.ksp_h, self.ksp_w).to(self.device)
        gan.get_summary()
        return gan

    def igtd_dset_prepare(self):
        self.data_handler.load_images()
        height, width = closest_factors(self.data_handler.get_n_features())
        if width > 15 or height > 15:
            raise ValueError(f"a dimension of the image is out of bound, height = {height}, width = {width}")

        self.ksp_h, self.ksp_w = KSP_LookUp.ksp(height, width)

    def gen_ae_model(self, ae_model):
        if 'igtd' in ae_model: self.igtd_dset_prepare()
        ae = AEHandler.AEHandler(ae_model, self.data_handler.get_n_features(), self.data_handler.get_n_classes(), ksp_h=self.ksp_h, ksp_w=self.ksp_w).to(self.device)
        ae.get_summary()
        return ae

    def read_data(self, location, target=None, title=None):
        self.title = f"{location.split('/')[-1].split('.')[0]}"
        if title is not None: self.title = self.title + "_" + title

        # Pipeline #1  reading data
        self.data_handler = DataHandler.DataHandler(self.batch_size, self.device, location, target)

    def train_ae(self, ae_model=None):
        if ae_model is None:
            ae_model = "no16" if self.data_handler.get_target() is None else "co16"

        self.ae = self.gen_ae_model(ae_model)

        img_loader = self.data_handler.get_img_loader() if 'igtd' in ae_model else None

        total_ae_loss = self.ae.train_model(self.data_handler.get_dataloader(), img_loader, epochs=1)

        HarryPlotter.plot_curve(f"Pre-trained AE Loss {self.title}", total_ae_loss)

        en_img = self.ae.encode(self.data_handler.get_feature_tensor(), self.data_handler.get_label_tensor())

        if img_loader is not None:
            img_tensor = self.data_handler.get_img_tensor()
            min_len = min(len(img_tensor), len(en_img))
            return torch.cat((img_tensor[:min_len], en_img[:min_len]), dim=0), torch.cat((self.data_handler.get_label_tensor(), self.data_handler.get_label_tensor()), dim=0)
        else:
            return en_img, self.data_handler.get_label_tensor()

    def train_gen_model(self, encoded_images, real_labels=None, gen_model=None):
        if gen_model is None:
            gen_model = "dc" if self.data_handler.get_target() is None else "dcc"

        dataset = TensorDataset(encoded_images, real_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.gen = self.gen_gen_model(gen_model)
        total_d_loss, total_g_loss = self.gen.train_model(dataloader, epochs=1)

        HarryPlotter.plot_curve(f"G & D Loss {self.title}", {"G": total_g_loss, "D": total_d_loss})

        return self.gen.sample(len(self.data_handler.get_dataframe())//self.data_handler.get_n_classes())

    def show_results(self, en_images, gen_images, en_labels=None, gen_labels=None):
        n_classes = self.data_handler.get_n_classes()

        fid_score = HarryPlotter.calculate_fid(en_images, gen_images, f'fid_{self.title}')
        print(f"FID score: {fid_score}")

        images, labels = filtering_samples(en_images, en_labels)
        HarryPlotter.plot_rgb_gigaplot(f"Enc IMGs {self.title}", images, labels, n_classes, fid_score)

        images, labels = filtering_samples(gen_images, gen_labels)
        HarryPlotter.plot_rgb_gigaplot(f"Gen IMGs {self.title}", images, labels, n_classes, fid_score)

        HarryPlotter.display_text_samples(f"5 real samples per classes {self.title}", self.data_handler.get_real_samples(self.data_handler.get_n_classes()*5))
        HarryPlotter.display_text_samples(f"5 fake samples per classes {self.title}", self.data_handler.get_fake_samples(self.ae, gen_images, gen_labels, self.data_handler.get_n_classes()*5))

    def save_full_synth_set(self, gen_images, gen_labels=None):
        dataframe = self.data_handler.get_fake_samples(self.ae, gen_images, gen_labels)
        dataframe.to_csv(f"results/synth_datasets/{self.title}.csv", index=False)
        return dataframe

    def multiple_sampling(self):
        # after training this function saves multiple samples of synthetic tabular data
        # if igtd inlcude encoded and igtd images
        if "igtd" in self.title:
            en_img = self.ae.encode(self.data_handler.get_feature_tensor(), self.data_handler.get_label_tensor())
            img_tensor = self.data_handler.get_img_tensor()
            min_len = min(len(img_tensor), len(en_img))
            all_real_img = torch.cat((img_tensor[:min_len], en_img[:min_len]), dim=0)
        else:
            all_real_img = self.ae.encode(self.data_handler.get_feature_tensor(), self.data_handler.get_label_tensor())

        original_title = self.title

        for i in range(5):
            gen_images, gen_labels = self.gen.sample(len(self.data_handler.get_dataframe()) // self.data_handler.get_n_classes())
            self.title = f"{self.title}_sample_{i}"
            HarryPlotter.calculate_fid(all_real_img, gen_images, f'fid_{self.title}')
            self.save_full_synth_set(gen_images, gen_labels)
            self.title = original_title

    def fit(self, source, title=None, target=None, ae_model=None, gen_model=None):
        # this function is a one-time callable function to run the entire framework
        # using one of the built-in GANs defined in the gan_model input
        # returns the synthetic dataset

        # step 1 read data
        self.read_data(source, target, title)
        # step 2 train ae

        en_images, en_labels = self.train_ae(ae_model=ae_model)
        # step 3 train built-in gen model
        gen_images, gen_labels = self.train_gen_model(en_images, en_labels, gen_model=gen_model)
        # step 4 show results
        self.show_results(en_images, gen_images, en_labels=en_labels, gen_labels=gen_labels)
        # step 5 save results
        return self.save_full_synth_set(gen_images, gen_labels)
