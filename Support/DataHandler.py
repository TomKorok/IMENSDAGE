import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scipy.io import arff


class DataHandler:
    def __init__(self, batch_size, device, location, dataset_title, target=None, classification=True):
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.device = device
        self.dataset_title = dataset_title
        self.target = target
        self.n_classes = None
        self.classification = classification
        self.n_features, self.dataframe, self.dataset, self.dataloader = self.read_data(location)
        # pandas dataset # Tensor dataset # tensor

    def read_data(self, location):
        # reading the data
        dataframe = read_file(location, self.target, self.classification)  # returning dataframe
        if self.classification:
            n_features = dataframe.shape[1] - 1
            # Normalize the features
            normalized_features = self.scaler.fit_transform(dataframe.iloc[:, :-1].values)
            # Convert to tensors, concat and then convert to dataloader
            features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=self.device)

            labels_tensor = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32, device=self.device)
            self.n_classes = torch.unique(labels_tensor).size()[0]
            dataset = TensorDataset(features_tensor, labels_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        else:
            n_features = dataframe.shape[1]
            # Normalize the features
            normalized_features = self.scaler.fit_transform(dataframe.values)
            # Convert to tensors, concat and then convert to dataloader
            features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=self.device)
            dataset = TensorDataset(features_tensor)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return n_features, dataframe, dataset, dataloader

    def get_n_features(self):
        return self.n_features

    def get_dataloader(self):
        return self.dataloader

    def get_dataframe(self):
        return self.dataframe

    def get_dataset_title(self):
        return self.dataset_title

    def get_features(self):
        return torch.stack([self.dataset[i][0] for i in range(len(self.dataset))])

    def get_labels(self):
        return torch.stack([self.dataset[i][1] for i in range(len(self.dataset))])

    def get_real_samples(self, amount):
        class_0 = self.dataframe[self.dataframe['Outcome'] == 0].iloc[:amount]
        class_1 = self.dataframe[self.dataframe['Outcome'] == 1].iloc[:amount]
        return pd.concat([class_0, class_1])

    def get_fake_samples(self, ae, gen_data, round_exceptions):
        fake_samples = np.array(ae.decode(gen_data["gen_images"], gen_data["labels"]).squeeze(dim=0).detach().cpu())

        # convert the samples to dataframe and redo the normalizing
        fake_samples = pd.DataFrame(fake_samples)
        fake_samples = self.scaler.inverse_transform(fake_samples)

        # concat the fake samples with its labels and convert to dataframe
        class_labels = np.expand_dims(gen_data["labels"].detach().cpu().numpy(), axis=1)
        if fake_samples.shape[1] != self.dataframe.shape[1]:
            fake_samples = np.concatenate((fake_samples, class_labels), axis=1)
        fake_samples = pd.DataFrame(fake_samples, columns=self.dataframe.columns)

        # round the necessary columns -- everything except for the exceptions
        fake_samples.loc[:, ~fake_samples.columns.isin(round_exceptions)] = fake_samples.loc[:,
                                                                            ~fake_samples.columns.isin(
                                                                                round_exceptions)].round(0)

        # set the column types equal to the ones in the original
        for column in fake_samples.columns:
            if column in self.dataframe:
                fake_samples[column] = fake_samples[column].astype(self.dataframe[column].dtype)

        return fake_samples

    def get_n_classes(self):
        return self.n_classes


def read_file(location, target, classification):
    if 'arff' in location:
        # Read the ARFF file
        data, meta = arff.loadarff(location)
        print("ARFF file loaded successfully!")
        # Convert the ARFF data to a pandas DataFrame
        df = pd.DataFrame(data)
    elif 'csv' in location:
        df = pd.read_csv(location)
        print("CSV file loaded successfully!")
    else:
        raise ValueError("Invalid input name provided")

    df = df.dropna()  # dropping uncompleted lines
    if classification:
        return handle_class(df, target)
    else:
        return df

def handle_class(df, target):
    type_column = df.pop(target)  # pop the target column
    df['Target'] = type_column  # move it to the end and rename it to target
    return df.astype(np.float32)