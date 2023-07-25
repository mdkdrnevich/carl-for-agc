from . import carl_utils

import uproot
import numpy as np
import torch
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
import awkward as ak
import collections

class AGCDataset:
    def __init__(
        self,
        features,
        data
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
        self.n_events = len(list(data.values())[0])
        self.array = data
        self.datas = []
        self.process()


    def process(self):
        """
        Handles conversion of dataset file at raw_path into Deep Sets dataset.

        """
        self.datas = []
        for i in range(self.n_events):
            total_data = []
            for feat in self.feature_names:
                #temp = []
                #for subfeat in self.features[feat]["subfeatures"]:
                    #print(subfeat)
                    #print(self.array[subfeat])
                #    temp.append(self.array[subfeat][i][None, :])
                #x = ak.concatenate(temp, axis=0)
                x = ak.concatenate([self.array[subfeat][i][None, :] for subfeat in self.features[feat]["subfeatures"]], axis=0)
                x = x.to_numpy().astype(np.float32).T
                x = torch.from_numpy(x)
                total_data.append(x[None, :])
            data = TensorDataset(*total_data)
            self.datas.append(data)

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.datas[idx]
    
    
def prep_inputs_for_eval(batch_list, features, x_scalers):
    num_features = len(features)
    sample_indices = [[0] for _ in range(num_features)]
    x_batch_list = [[] for _ in range(num_features)]
    for sample in batch_list:
        for i, t in enumerate(sample[:num_features]):
            # Empty vectors (i.e. sets) will cause nans in the model, so we put a single zero vector instead
            if t.size(dim=0) == 0:
                t = torch.zeros(1, features[list(features.keys())[i]]["size"])
            x_batch_list[i].append(t)
            sample_indices[i].append(t.size(dim=0))
    x_batch = []
    for i, feat in enumerate(features):
        x = torch.cat(x_batch_list[i], dim=0)
        x = x_scalers[i].transform(x).astype(np.float32)
        if features[feat]["set"] is True:
            x_batch.append(torch.from_numpy(x.T)[None, :]) # Model expects rows to be features
        else:
            x_batch.append(torch.from_numpy(x)[None, :])
    sample_indices = torch.from_numpy(np.array([np.cumsum(s) for s in sample_indices]))
    return (x_batch,), sample_indices


def get_eval_DataLoader(generator, features, scalers, batch_size=128, shuffle=False):
    features = collections.OrderedDict(sorted(features.items()))
    loader = DataLoader(generator, batch_size=batch_size, shuffle=shuffle)
    loader.collate_fn = lambda batch: prep_inputs_for_eval(batch, features, scalers)
    return loader


def get_feature_map(jets, electrons, muons):
    carl_feature_map = {
        "Jet_pt": jets.pt,
        "Jet_eta": jets.eta,
        "Jet_phi": jets.phi,
        "Jet_mass": jets.mass,
        "Jet_btagCSVV2": jets.btagCSVV2,
        "Electron_pt": electrons.pt,
        "Electron_eta": electrons.eta,
        "Electron_phi": electrons.phi,
        "Electron_mass": electrons.mass,
        "Muon_pt": muons.pt,
        "Muon_eta": muons.eta,
        "Muon_phi": muons.phi,
        "Muon_mass": muons.mass
    }
    return carl_feature_map


def get_inference_results_local(model, features, feature_map, X_scalers):
    ds = AGCDataset(features, feature_map)
    loader = get_eval_DataLoader(carl_utils.preprocessing.ConcatDataset(ds), features, X_scalers)
    carl_weights = carl_utils.eval.get_r_hats(model, loader)
    return carl_weights