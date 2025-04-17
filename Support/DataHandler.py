import os
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sympy.codegen.ast import float32
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scipy.io import arff
import csv

from Support import CustomEncoder


def is_removable(col):
    return 'id' in col.lower() or col.strip() == '' or 'unnamed' in col.lower() or '#' in col.lower()


class DataHandler:
    def __init__(self, batch_size, device, location, target=None):
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.device = device
        self.target = target
        self.n_classes = None
        self.class_labels = None
        self.location = location
        self.label_encoder = CustomEncoder.CustomEncoder()
        self.encoded_columns = None
        self.dropped_columns = None
        self.binary_columns = None
        self.img_tensor = None
        self.img_loader = None
        self.n_features, self.dataframe, self.dataset, self.dataloader = self.read_data()
        # pandas dataset # Tensor dataset # tensor

    def read_data(self):
        # this function receives the dataframe from read_file
        # after it, it handles normalization and creating the dataloader
        dataframe = self.read_file()  # returning dataframe

        n_features = dataframe.shape[1] - 1
        # Normalize the features
        normalized_features = self.scaler.fit_transform(dataframe.iloc[:, :-1].values)
        # Convert to tensors, concat and then convert to dataloader
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=self.device)

        labels_tensor = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32, device=self.device)
        self.n_classes = torch.unique(labels_tensor).size()[0]
        self.class_labels = np.sort(dataframe['Target'].unique())
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        return n_features, dataframe, dataset, dataloader

    def get_n_features(self):
        return self.n_features

    def get_dataloader(self):
        return self.dataloader

    def get_dataframe(self):
        return self.dataframe.drop(columns=['Target']) if self.target is None else self.dataframe

    def get_feature_tensor(self):
        return torch.stack([self.dataset[i][0] for i in range(len(self.dataset))])

    def get_img_loader(self):
        return self.img_loader

    def get_img_tensor(self):
        return self.img_tensor

    def get_target(self):
        return self.target

    def reattach_dropped_columns(self, dataframe):
        min_len = min(len(self.dropped_columns), len(dataframe))
        dropped = self.dropped_columns.iloc[:min_len].reset_index(drop=True)
        fake = dataframe.iloc[:min_len].reset_index(drop=True)
        return pd.concat([dropped, fake], axis=1)

    def get_label_tensor(self):
        return torch.stack([self.dataset[i][1] for i in range(len(self.dataset))])

    def decode_columns(self, dataframe):
        dataframe[self.encoded_columns] = self.label_encoder.inverse_transform(dataframe[self.encoded_columns])
        return dataframe

    def set_header_dtype(self, dataframe):
        for column in dataframe.columns:
            if column in self.dataframe:
                dataframe[column] = dataframe[column].astype(self.dataframe[column].dtype)
        return dataframe

    def sample_amount(self, amount, dataframe):
        #sample amount of each class and decode the label encoding
        dataframe = self.set_header_dtype(dataframe)
        if self.target is not None:
            selected_classes = [dataframe[dataframe['Target'] == label].iloc[:amount//len(self.class_labels)] for label in self.class_labels]
            r_dataframe = pd.concat(selected_classes)
            if self.label_encoder.get_fitted():
                r_dataframe = self.decode_columns(r_dataframe)
            return r_dataframe.rename(columns={'Target': self.target})
        else:
            if self.label_encoder.get_fitted():
                dataframe = self.decode_columns(dataframe)
            dataframe = dataframe.drop(columns=['Target'])

        return dataframe.iloc[:amount]

    def get_real_samples(self, amount):
        return self.sample_amount(amount, self.dataframe.copy())

    def get_fake_samples(self, ae, gen_images, gen_labels=None, amount=None):
        is_reattach = True if amount is None and self.dropped_columns is not None else False
        if amount is None: amount = gen_images.shape[0]
        fake_samples = np.array(ae.decode(gen_images, gen_labels).squeeze(dim=0).detach().cpu())

        # convert the samples to dataframe and redo the normalizing
        fake_samples = pd.DataFrame(fake_samples)
        fake_samples = self.scaler.inverse_transform(fake_samples)

        # concat the fake samples with its labels and convert to dataframe
        class_labels = np.expand_dims(gen_labels.detach().cpu().numpy(), axis=1)
        if fake_samples.shape[1] != self.dataframe.shape[1]:
            fake_samples = np.concatenate((fake_samples, class_labels), axis=1)
        fake_samples = pd.DataFrame(fake_samples, columns=self.dataframe.columns)

        fake_samples = self.sample_amount(amount, fake_samples)
        if is_reattach:
            fake_samples = self.reattach_dropped_columns(fake_samples)
        return fake_samples

    def get_n_classes(self):
        return self.n_classes

    def get_location(self):
        return self.location

    def is_classification(self):
        return True if self.target is not None else False

    def detect_delimiter(self):
        with open(self.location, 'r', newline='', encoding='utf-8') as file:
            sample = file.read(50)  # Read a small portion of the file
            sniffer = csv.Sniffer()
            delimiter = sniffer.sniff(sample).delimiter
            return delimiter


    def remove_unnec_cols(self, df):
        cols_to_remove = [col for col in df.columns if is_removable(col)]
        if cols_to_remove:
            self.dropped_columns = df[cols_to_remove]
            df = df.drop(columns=cols_to_remove)
        return df

    def set_target(self, df):
        try:
            label_column = df.pop(self.target) # pop the target column
            df['Target'] = label_column # move it to the end
        except Exception:
            df['Target'] = 0
        return df

    def detect_bin_col(self, df):
        self.binary_columns = [col for col in df.columns if df[col].dropna().nunique() == 2]

    def read_file(self):
        # this method is reading the data and setting the target column
        # it will add a Target column in the end... if the dataset is non-classification it will only contain 0
        if 'arff' in self.location:
            # Read the ARFF file
            data, meta = arff.loadarff(self.location)
            print("ARFF file loaded successfully!")
            # Convert the ARFF data to a pandas DataFrame
            df = pd.DataFrame(data)
        elif 'csv' in self.location:
            df = pd.read_csv(self.location, sep=self.detect_delimiter())
            print("CSV file loaded successfully!")
        else:
            raise ValueError("Invalid input name provided")

        df = df.dropna()  # dropping uncompleted lines
        df = self.remove_unnec_cols(df) #drops id columns
        df = self.set_target(df) # handles and rename the target col if any else adds one with 0
        df = self.cat_encoding(df) # label encode the class labels as well
        self.detect_bin_col(df)
        return df

    def cat_encoding(self, df):
        # Threshold for uniqueness
        threshold = 0.05 * len(df)

        # Find columns where unique values are below threshold df[col].nunique() < threshold and
        self.encoded_columns = [col for col in df.columns if df[col].dtype == 'object']

        if self.encoded_columns:
            # Convert all selected columns to strings, then apply lowercasing and stripping
            df[self.encoded_columns] = df[self.encoded_columns].astype(str).apply(lambda col: col.str.lower().str.strip())
            # label encoding
            df[self.encoded_columns] = self.label_encoder.fit_transform(df[self.encoded_columns])

        return df

    def get_path_to_img(self):
        path = f"source/images/{self.location.split('/')[-1].split('.')[0]}"
        if self.target is not None:
            path = f"{path}/only_feature"
        else:
            if os.path.exists(f"{path}/label_as_feature"):
                path = f"{path}/label_as_feature"
            else:
                pass
        return path

    def rename_txt(self, path):
        files = os.listdir(path)

        # Loop through each file
        for file_name in files:
            # Check if it's a .txt file
            if file_name.endswith('.txt') and file_name[0] == '_':
                    # Create the new file name by removing the first character
                    new_name = file_name[1:].split('_')[0]
                    new_name = new_name.zfill(6)
                    new_name = f"{new_name}.txt"
                    # Get the full file paths
                    old_file_path = os.path.join(path, file_name)
                    new_file_path = os.path.join(path, new_name)
                    # Rename the file
                    os.rename(old_file_path, new_file_path)

    def load_images(self):
        path = self.get_path_to_img()
        self.rename_txt(path)
        # Get all txt files in the folder
        txt_files = [f for f in os.listdir(path) if f.endswith(".txt")]

        # List to store tensors
        tensors = []

        for file in txt_files:
            file_path = f"{path}/{file}"

            # Load the matrix from txt
            matrix = torch.tensor(
                [list(map(lambda x: float(x), line.split())) for line in open(file_path)],
                dtype=torch.float32
            )

            matrix /= 255.0

            tensors.append(matrix)

        self.img_tensor = torch.stack(tensors).unsqueeze(1).to(self.device)

        dataset = TensorDataset(self.img_tensor, self.get_label_tensor())
        self.img_loader = DataLoader(dataset, batch_size=self.batch_size)