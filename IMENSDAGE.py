import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scipy.io import arff

import GAN
import AE

#IMage ENcoding for Synthetic DAta GEnaration

def read_file(location):
    if 'arff' in location:
        # Read the ARFF file
        data, meta = arff.loadarff(location)
        # Convert the ARFF data to a pandas DataFrame
        df = pd.DataFrame(data)
        df = df.dropna() # dropping uncompleted lines
        type_column = df.pop('TYPE')
        df['Outcome'] = type_column
        df = df.astype(np.float32)
    elif 'csv' in location:
        df = pd.read_csv(location)
    else:
        raise ValueError("Invalid input name provided")

    return df


def plot_gray_gigaplot(images, title, num_classes=10, examples_per_class=10):
    images = (images + 1) / 2
    fig, axes = plt.subplots(num_classes, examples_per_class, figsize=(examples_per_class, num_classes))
    for i in range(num_classes):
        for j in range(examples_per_class):
            idx = i * examples_per_class + j
            img = images[idx].detach().cpu().squeeze().numpy()  # Convert to numpy
            axes[i, j].imshow(img, cmap='gray')  # Assuming grayscale images
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=1)
    plt.savefig(f'results/generated_images/{title}.png')
    plt.show()


def plot_curve(title, array):
    curve = np.convolve(array, np.ones((1,)) / 1, mode='valid')
    plt.plot([j for j in range(len(curve))], curve, color='darkorange', alpha=1)
    plt.title(title)
    plt.ylabel("Loss")
    plt.xlabel("Steps")
    plt.savefig(f"results/curves/{title}.png")
    plt.show()


def print_default():
    # getting and setting and displaying the used device
    try:
        print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"ID of current CUDA device:{torch.cuda.current_device()}")
        print(f"Name of current CUDA device:{torch.cuda.get_device_name(torch.cuda.current_device())}")
    except Exception as e:
        print(e)

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IMENSDAGE:
    def __init__(self, batch_size=128, n_classes=2):
        self.device = print_default()
        self.scaler = StandardScaler()
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.n_features = None
        self.dataloader = None # tensor dataloader
        self.dataframe = None # pandas dataset
        self.dataset = None # Tensor dataset

    def plot_rgb_gigaplot(self, title, images):
        examples_per_class = int(images.shape[0] / 2)
        # Scale from [-1, 1] to [0, 255]
        images = ((images + 1) * 127.5).clip(0, 255).to(torch.uint8)

        fig, axes = plt.subplots(self.n_classes, examples_per_class, figsize=(examples_per_class, self.n_classes))
        for i in range(self.n_classes):
            for j in range(examples_per_class):
                idx = i * examples_per_class + j
                img = images[idx].detach().cpu().permute(1, 2, 0).numpy()  # Convert to numpy and rearrange channels
                axes[i, j].imshow(img)  # No cmap, since it's color
                axes[i, j].axis('off')
        plt.tight_layout()
        plt.suptitle(title, fontsize=12, y=1)
        plt.savefig(f"results/generated_images/{title}.png")
        plt.show()

    def read_data(self, location, conditional):
        # reading the data
        try:
            self.dataframe = read_file(location) # returning dataframe
        except ValueError as e:
            print(e)
        self.n_features = (self.dataframe.shape[1] - 1) if conditional else self.dataframe.shape[1]

        if conditional:
            # Normalize the features
            normalized_features = self.scaler.fit_transform(self.dataframe.iloc[:, :-1].values)
            # Convert to tensors, concat and then convert to dataloader
            features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=self.device)
        else:
            # normalize the labels as well
            normalized_features = self.scaler.fit_transform(self.dataframe.values)
            features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=self.device)

        labels_tensor = torch.tensor(self.dataframe.iloc[:, -1].values, dtype=torch.float32, device=self.device)
        self.dataset = TensorDataset(features_tensor, labels_tensor)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def get_real_samples(self, autoencoder):
        # Select first 5 elements where label is 0 and the first 5 where label is 1
        class_0 = self.dataframe[self.dataframe['Outcome'] == 0].iloc[:5]
        class_1 = self.dataframe[self.dataframe['Outcome'] == 1].iloc[:5]
        combined = pd.concat([class_0, class_1])
        if autoencoder.get_conditional():
            labels = combined.pop('Outcome')
        else:
            labels = combined.pop('Outcome')
            combined["Outcome"] = labels

        combined_t = torch.tensor(combined.values, dtype=torch.float32, device=self.device)
        labels_t = torch.tensor(labels.values, dtype=torch.float32, device=self.device)
        # encode only features with labels
        if autoencoder.get_conditional():
            combined["Outcome"] = labels
        return combined, autoencoder.encode(combined_t, labels_t)

    def display_text_samples(self, title, dataframe):
        print("")
        print(f"{title}:")
        print(dataframe.head(len(dataframe)).to_string())

    def generate_result(self, gan, autoencoder, round_exceptions, title, conditional):
        # generate and display 5 real examples + the encoded images from both classes
        real_samples, encoded_images = self.get_real_samples(autoencoder)
        self.display_text_samples(f"5-5 Real Samples {title} C={conditional}", real_samples)
        self.plot_rgb_gigaplot(f"Encoded Images {title} C={conditional}", encoded_images)

        # generate and display 5 fake examples + the generated images from both classes
        fake_samples, generated_fake_images = gan.sample(350, self.dataframe.head(5), autoencoder, self.scaler, round_exceptions)
        self.display_text_samples(f"5-5 Fake Samples {title} C={conditional}", pd.concat([fake_samples.iloc[:5], fake_samples.iloc[350:355]]))
        self.plot_rgb_gigaplot(f"Generated Images {title} C={conditional}", torch.cat((generated_fake_images[:5, :, :, :], generated_fake_images[350:355, :, :, :]), dim=0))
        return fake_samples

    def generate_models(self, conditional):
        autoencoder = AE.Autoencoder(conditional, self.n_features, self.n_classes).to(self.device)
        gan = GAN.GAN(self.batch_size, 28, 7, 7, 2, conditional=conditional).to(self.device)

        # print the summary of the AE and G
        autoencoder.get_summary()
        gan.get_summary()
        return gan, autoencoder

    def train_models(self, location, title, conditional):
        self.read_data(location, conditional)
        gan, autoencoder = self.generate_models(conditional)
        # pre-train the encoder
        total_ae_loss = autoencoder.train_model(self.dataloader, epochs=500)
        plot_curve(f"Pre-trained AE Loss {title} C={conditional}", total_ae_loss)

        # gan training -- this also plots the training curves
        total_d_loss, total_g_loss = gan.train_model(autoencoder, self.dataloader, epochs=5000)
        plot_curve(f"Discriminator Loss {title} C={conditional}", total_d_loss)
        plot_curve(f"Generator Loss {title} C={conditional}", total_g_loss)

        return gan, autoencoder

    def fit(self, location, round_exceptions=None, title=None, conditional=True):
        gan, autoencoder = self.train_models(location, title, conditional)
        return self.generate_result(gan, autoencoder, round_exceptions, title, conditional)