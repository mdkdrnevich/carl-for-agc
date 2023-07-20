import sys
#sys.path.append('/home/mdd424/CARL_tthbb/')
#sys.path.append('/home/mdd424/downloads/carl-torch')

import uproot
import numpy as np
import math
import json
import bisect
import os
import pickle
import logging
#from tqdm.notebook import tqdm

import torch
from torch import nn
from torch import sigmoid
from torch.utils.data import DataLoader
#from torchsummary import summary

from sklearn import preprocessing
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity

from scipy.spatial import distance
import awkward as ak

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE

jet_features = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"]
electron_features = ["Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass"]
muon_features = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"]

features = jet_features + electron_features + muon_features

#weight_features = ["weight_mc", "weight_pileup", "weight_leptonSF", "weight_jvt", "weight_bTagSF_DL1r_Continuous"]
weight_features = ["genWeight", "btagWeight_CSVV2"]

# Luminosities in pb^-1
luminosities = {'2015': 36207.66, '2017': 44307.4, '2018': 58450.1}
luminosities_by_run = {'9364': 36207.66, '10201': 44307.4, '10724': 58450.1}

with open("/home/mdrnevich/AGC/carl-for-agc/nanoaod_inputs.json", 'r') as f:
    file_dict = json.load(f)

data_metadata_dict = {}

all_nominal_files = [x["path"] for x in file_dict["ttbar"]["nominal"]["files"]]
print(len(all_nominal_files), all_nominal_files[0])
np.random.shuffle(all_nominal_files)
all_nominal_files[0]

all_variation_files = [x["path"] for x in file_dict["ttbar"]["PS_var"]["files"]]
print(len(all_variation_files), all_variation_files[0])
np.random.shuffle(all_variation_files)
all_variation_files[0]

with open("/home/mdrnevich/AGC/carl-for-agc/carl_data_metadata.json", 'r') as f:
    data_metadata_dict = json.load(f)
    max_data_index = data_metadata_dict["max_data_index"]
    max_jet_size = data_metadata_dict["max_jet_size"]
    max_electron_size = data_metadata_dict["max_electron_size"]
    max_muon_size = data_metadata_dict["max_muon_size"]

train_frac = 0.6
val_frac = 0.2
test_frac = 0.2

max_train_index = int(train_frac * max_data_index)
max_val_index = max_train_index + int(val_frac * max_data_index)
max_test_index = max_val_index + int(test_frac * max_data_index)

np.random.shuffle(all_nominal_files)
np.random.shuffle(all_variation_files)
print(max_train_index, max_val_index, max_test_index)

def get_max_sizes(filename):
    dataset = uproot.open(filename)["Events"].arrays([jet_features[0], electron_features[0], muon_features[0]])
    max_jet_size = max(map(len, dataset[jet_features[0]]))
    max_electron_size = max(map(len, dataset[electron_features[0]]))
    max_muon_size = max(map(len, dataset[muon_features[0]]))
    return [max_jet_size, max_electron_size, max_muon_size]

def fill_or_extend_tree(datafile, tree_dict, treename="Events"):
    if datafile.get(treename) is None:
        datafile[treename] = tree_dict
    else:
        datafile[treename].extend(tree_dict)
    return None

def build_data_dict(features, arrays, split_index=None, split_low=False, split_high=False):
    if split_low is False and split_high is False:
        data_dict = dict(zip(features, ak.unzip(arrays)))
    elif split_high is True:
        data_dict = dict(zip(features, ak.unzip(arrays[:split_index])))
    elif split_low is True:
        data_dict = dict(zip(features, ak.unzip(arrays[split_index:])))
    return data_dict

df_train = uproot.recreate("/data/mdrnevich/AGC/CMS_ttbar_nominal_DeepSets_training_data.root")
df_val = uproot.recreate("/data/mdrnevich/AGC/CMS_ttbar_nominal_DeepSets_validation_data.root")
df_test = uproot.recreate("/data/mdrnevich/AGC/CMS_ttbar_nominal_DeepSets_testing_data.root")

chunk_size = 100000

current_index = 0
for filename in all_nominal_files:
    print("Loading file: {}".format(filename))
    # Load the data
    nominal_dataset = uproot.open(filename)["Events"]
    filesize = int(nominal_dataset.arrays(jet_features[0]).type.length)
    print(filesize)
    for i in range(int(np.ceil(filesize / chunk_size))):
        jet_arr = nominal_dataset.arrays(jet_features, entry_start=int(i * chunk_size), entry_stop=int((i+1) * chunk_size))
        electron_arr = nominal_dataset.arrays(electron_features, entry_start=int(i * chunk_size), entry_stop=int((i+1) * chunk_size))
        muon_arr = nominal_dataset.arrays(muon_features, entry_start=int(i * chunk_size), entry_stop=int((i+1) * chunk_size))
        weight_arr = nominal_dataset.arrays(weight_features, entry_start=int(i * chunk_size), entry_stop=int((i+1) * chunk_size))
        # Get the run number
        #nominal_file_run_number = str(NOMINAL_FILE_TO_RUN_NUMBER[filename])
        # Get the DSID number
        #nominal_file_dsid = str(NOMINAL_FILE_TO_DSID[filename])
        # Get the luminsotiy, DSID cross section, and per DSID total weighted events
        #_scale_factor = luminosities_by_run[nominal_file_run_number] * NOMINAL_XSECTIONS[nominal_file_dsid] / NOMINAL_NORMALIZATIONS[nominal_file_dsid]
        _scale_factor = 3378 * 1 / 1
        # Extract the combined weight array
        _weights = ak.concatenate(ak.unzip(weight_arr[weight_features][:, np.newaxis]), axis=1).to_numpy().prod(axis=1)

        current_data_size = len(_weights)
        
        jet_sets = []
        electron_sets = []
        muon_sets = []
        for i in range(current_data_size):
            jet_sets.append(
                np.concatenate([x.to_numpy()[:, np.newaxis] for x in ak.unzip(jet_arr[jet_features][i])], axis=1).flatten()
            )
            electron_sets.append(
                np.concatenate([x.to_numpy()[:, np.newaxis] for x in ak.unzip(electron_arr[electron_features][i])], axis=1).flatten()
            )
            muon_sets.append(
                np.concatenate([x.to_numpy()[:, np.newaxis] for x in ak.unzip(muon_arr[muon_features][i])], axis=1).flatten()
            )

        nominal_features = ["jet_4vec", "electron_4vec", "muon_4vec"]
        nominal_arr = ak.Array({"jet_4vec": jet_sets,
                                "electron_4vec": electron_sets,
                                "muon_4vec": muon_sets})

        # put everything in train
        if current_index + current_data_size < max_train_index:
            data_dict = build_data_dict(nominal_features, nominal_arr)
            data_dict["weight_mc_combined"] = _weights * _scale_factor
            fill_or_extend_tree(df_train, data_dict)
        # put part in train and the rest in val
        elif current_index < max_train_index and current_index + current_data_size < max_val_index:
            split_index = max_train_index - current_index
            data_dict = build_data_dict(nominal_features, nominal_arr, split_index=split_index, split_high=True)
            data_dict["weight_mc_combined"] = _weights[:split_index] * _scale_factor
            fill_or_extend_tree(df_train, data_dict)

            data_dict = build_data_dict(nominal_features, nominal_arr, split_index=split_index, split_low=True)
            data_dict["weight_mc_combined"] = _weights[split_index:] * _scale_factor
            fill_or_extend_tree(df_val, data_dict)
        # put everything into val
        elif current_index >= max_train_index and current_index + current_data_size < max_val_index:
            data_dict = build_data_dict(nominal_features, nominal_arr)
            data_dict["weight_mc_combined"] = _weights * _scale_factor
            fill_or_extend_tree(df_val, data_dict)
        # put part in val and the rest in test
        elif current_index < max_val_index and current_index + current_data_size < max_test_index:
            split_index = max_val_index - current_index
            data_dict = build_data_dict(nominal_features, nominal_arr, split_index=split_index, split_high=True)
            data_dict["weight_mc_combined"] = _weights[:split_index] * _scale_factor
            fill_or_extend_tree(df_val, data_dict)

            data_dict = build_data_dict(nominal_features, nominal_arr, split_index=split_index, split_low=True)
            data_dict["weight_mc_combined"] = _weights[split_index:] * _scale_factor
            fill_or_extend_tree(df_test, data_dict)
        # put everything into test
        elif current_index >= max_val_index and current_index + current_data_size < max_test_index:
            data_dict = build_data_dict(nominal_features, nominal_arr)
            data_dict["weight_mc_combined"] = _weights * _scale_factor
            fill_or_extend_tree(df_test, data_dict)
        # put what's needed into test and ignore the rest
        elif current_index < max_test_index and current_index + current_data_size >= max_test_index:
            split_index = max_test_index - current_index
            data_dict = build_data_dict(nominal_features, nominal_arr, split_index=split_index, split_high=True)
            data_dict["weight_mc_combined"] = _weights[:split_index] * _scale_factor
            fill_or_extend_tree(df_test, data_dict)
        else:
            print("Uncaught case:",
                  "current_index: {}".format(current_index),
                  "current_data_size: {}".format(current_data_size),
                  "max_train_index: {}".format(max_train_index),
                  "max_val_index: {}".format(max_val_index),
                  "max_test_index: {}".format(max_test_index), sep='\n')

        current_index += current_data_size
        if current_index >= max_test_index:
            break
    if current_index >= max_test_index:
        break
print("Finished!")