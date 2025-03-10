import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from scipy.io import arff


class DataHandler:
    def __init__(self, batch_size, device, location, round_exceptions, dataset_title, target=None):
        self.scaler = StandardScaler()
        self.batch_size = batch_size
        self.device = device
        self.round_exceptions = round_exceptions
        self.dataset_title = dataset_title
        self.target = target
        self.n_classes = None
        self.class_labels = None
        self.location = location
        self.n_features, self.dataframe, self.dataset, self.dataloader = self.read_data()
        # pandas dataset # Tensor dataset # tensor

    def read_data(self):
        # reading the data
        dataframe = read_file(self.location, self.target)  # returning dataframe
        dataframe = dataframe.sort_values(by=dataframe.columns[-1])

        n_features = dataframe.shape[1] - 1
        # Normalize the features
        normalized_features = self.scaler.fit_transform(dataframe.iloc[:, :-1].values)
        # Convert to tensors, concat and then convert to dataloader
        features_tensor = torch.tensor(normalized_features, dtype=torch.float32, device=self.device)

        labels_tensor = torch.tensor(dataframe.iloc[:, -1].values, dtype=torch.float32, device=self.device)
        self.n_classes = torch.unique(labels_tensor).size()[0]
        self.class_labels = np.sort(dataframe['Target'].unique())
        dataset = TensorDataset(features_tensor, labels_tensor)
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

    def get_feature_tensor(self):
        return torch.stack([self.dataset[i][0] for i in range(len(self.dataset))])

    def get_label_tensor(self):
        return torch.stack([self.dataset[i][1] for i in range(len(self.dataset))])

    def get_real_samples(self, amount, dataframe=None):
        if dataframe is None:
            dataframe = self.dataframe
        try:
            selected_classes = [dataframe[dataframe['Target'] == label].iloc[:amount//len(self.class_labels)] for label in self.class_labels]
            r_dataframe = pd.concat(selected_classes)
            if self.target is None:
                return r_dataframe.drop(columns=['Target'])
            else:
                return r_dataframe
        except Exception:
            pass
        if self.target is None:
            dataframe = dataframe.drop(columns=['Target'])
        return dataframe.iloc[:amount]

    def get_fake_samples(self, ae, gen_images, gen_labels=None, amount=None):
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

        # round the necessary columns -- everything except for the exceptions
        fake_samples.loc[:, ~fake_samples.columns.isin(self.round_exceptions)] = fake_samples.loc[:,
                                                                            ~fake_samples.columns.isin(
                                                                                self.round_exceptions)].round(0)

        # set the column types equal to the ones in the original
        for column in fake_samples.columns:
            if column in self.dataframe:
                fake_samples[column] = fake_samples[column].astype(self.dataframe[column].dtype)

        return self.get_real_samples(amount, fake_samples)

    def get_n_classes(self):
        return self.n_classes

    def get_title(self):
        return self.dataset_title

    def get_location(self):
        return self.location

    def is_classification(self):
        return True if self.target is not None else False


def read_file(location, target):
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
    try:
        label_column = df.pop(target) # pop the target column
        df['Target'] = label_column # move it to the end and rename it to target
    except Exception:
        df['Target'] = 0

    return df.astype(np.float32)

