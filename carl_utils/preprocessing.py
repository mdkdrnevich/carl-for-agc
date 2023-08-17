import uproot
import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from sklearn import preprocessing
import collections
import zipfile
import yaml
from tqdm import tqdm

# DataSets class that can process any data formatted according the standard features configurations
class DeepSetsDataset:
    def __init__(
        self,
        file_names,
        features,
        label,
        weights="weight_mc_combined",
        start_event=0,
        stop_event=1000,
        npad=0,
    ):
        """
        Initialize parameters of Deep Sets dataset
        Args:
            root (str): path
            n_events (int): how many events to process (-1=all)
            n_events_merge (int): how many events to merge
            file_names (list of strings): file names
            remove_unlabeled (boolean): remove unlabeled data samples
        """
        self.features = collections.OrderedDict(sorted(features.items()))
        self.feature_names = list(self.features.keys())
        self.label = label
        self.weights = weights
        self.n_events = stop_event - start_event
        self.start_event = start_event
        self.stop_event = stop_event
        self.file_names = file_names
        self.npad = npad
        self.datas = []
        self.process()


    def process(self):
        """
        Handles conversion of dataset file at raw_path into Deep Sets dataset.

        """
        self.datas = []
        for raw_path in self.file_names:
            with uproot.open(raw_path) as root_file:

                tree = root_file["Events"]

                feature_array = tree.arrays(
                    self.feature_names,
                    entry_start=self.start_event,
                    entry_stop=self.stop_event,
                    library="ak",
                )

                weight_array = tree.arrays(
                    self.weights,
                    entry_start=self.start_event,
                    entry_stop=self.stop_event,
                    library='ak',
                )

            if self.label == 0:
                y = np.zeros((self.n_events, 1))
            elif self.label == 1:
                y = np.ones((self.n_events, 1))

            for i in tqdm(range(self.n_events)):
                total_data = []
                for feat in self.feature_names:
                    x = feature_array[feat][i].to_numpy().astype(np.float32)
                    if self.features[feat]["set"] is True:
                        # Each event is a flattened array with entries [feat1, feat2, feat3, ..., featn] * n_particles
                        # 'x' is a 2D tensor with shape (M,N) where N is the number of features and M is the number of particles
                        x = x.reshape(-1, self.features[feat]["size"])
                    else:
                        x = x.reshape(1, -1)
                    x = torch.from_numpy(x)
                    total_data.append(x[None, :])

                Y = torch.tensor(y[i : i + 1], dtype=torch.float)
                w = torch.tensor([[weight_array[self.weights][i]]], dtype=torch.float)
                data = TensorDataset(*total_data, Y, w)
                self.datas.append(data)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.datas[idx]


def CombinedDataset(*args):
    total = []
    for arg in args:
        total.extend(arg.datas)
    return ConcatDataset(total)


def prep_inputs_for_scaling(batch_list):
    num_features = len(batch_list[0]) - 2
    x_batch_list = [[] for i in range(num_features)]
    w_batch_list = []
    for s in batch_list:
        for i, t in enumerate(s[:num_features]):
            x_batch_list[i].append(t)
        w_batch_list.append(s[-1])
    x_batch = []
    for i in range(num_features):
        x_batch.append(torch.cat(x_batch_list[i], dim=0))
    w_batch = torch.cat(w_batch_list)
    return x_batch, w_batch


def get_scaling(dataset, features, batch_size=1024, shuffle=False):
    scaling_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    scaling_loader.collate_fn = prep_inputs_for_scaling

    num_features = len(features)
    X_scalers = [preprocessing.StandardScaler() for _ in range(num_features)]
    weight_total = 0
    num_total_weights = 0
    
    for batch in tqdm(scaling_loader):
        for i in range(num_features):
            X_scalers[i].partial_fit(batch[0][i])
        weight_total += batch[1].sum()
        num_total_weights += len(batch[1])
    weight_norm = weight_total / num_total_weights
    return X_scalers, weight_norm


def prep_inputs_for_training(batch_list, features, x_scalers, weight_norm=1):
    num_features = len(features)
    sample_indices = [[] for _ in range(num_features)]
    x_batch_list = [[] for _ in range(num_features)]
    y_batch_list = []
    w_batch_list = []
    for sample in batch_list:
        for i, t in enumerate(sample[:num_features]):
            # Empty vectors (i.e. sets) will cause nans in the model, so we put a single zero vector instead
            if t.size(dim=0) == 0:
                t = torch.zeros(1, features[list(features.keys())[i]]["size"])
            x_batch_list[i].append(t)
            sample_indices[i].append(t.size(dim=0))
        y_batch_list.append(sample[-2])
        w_batch_list.append(sample[-1])
    x_batch = []
    for i, feat in enumerate(features):
        x = torch.cat(x_batch_list[i], dim=0)
        x = x_scalers[i].transform(x).astype(np.float32)
        if features[feat]["set"] is True:
            x_batch.append(torch.from_numpy(x.T)[None, :]) # Model expects rows to be features
        else:
            x_batch.append(torch.from_numpy(x)[None, :])
    y_batch = torch.cat(y_batch_list, dim=0)
    w_batch = torch.cat(w_batch_list, dim=0) / weight_norm
    sample_indices = torch.tensor(sample_indices)
    #sample_indices = torch.from_numpy(np.array([np.cumsum(s) for s in sample_indices]))
    return (x_batch, y_batch[:, None], w_batch[:, None]), sample_indices


def get_training_DataLoader(generator, features, scalers, weight_norm=1, batch_size=128, shuffle=True):
    features = collections.OrderedDict(sorted(features.items()))
    loader = DataLoader(generator, batch_size=batch_size, shuffle=shuffle)
    loader.collate_fn = lambda batch: prep_inputs_for_training(batch, features, scalers, weight_norm=weight_norm)
    return loader


def load_scaling(path_to_zip):
    with zipfile.ZipFile(path_to_zip, 'r') as zf:
        scaling_metadata = yaml.load(zf.read("deepsets_metadata.yaml"), Loader=yaml.CLoader)["scaling"]
    X_scalers = []
    for input in scaling_metadata["inputs"]:
        scaler = preprocessing.StandardScaler()
        scaler.mean_ = input["mean"]
        scaler.scale_ = input["scale"]
        scaler.var_ = input["var"]
        X_scalers.append(scaler)
    return X_scalers, scaling_metadata["weights"]