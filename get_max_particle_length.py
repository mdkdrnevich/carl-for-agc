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

import torch
from torch import nn
from torch import sigmoid
from torch.utils.data import DataLoader

from sklearn import preprocessing
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KernelDensity

from scipy.spatial import distance
import awkward as ak


if __name__ == "__main__":
    jet_features = ["Jet_pt", "Jet_eta", "Jet_phi", "Jet_mass"]
    electron_features = ["Electron_pt", "Electron_eta", "Electron_phi", "Electron_mass"]
    muon_features = ["Muon_pt", "Muon_eta", "Muon_phi", "Muon_mass"]
    
    with open("nanoaod_inputs.json", 'r') as f:
        file_dict = json.load(f)
        
    all_nominal_files = [x["path"] for x in file_dict["ttbar"]["nominal"]["files"]]
    np.random.shuffle(all_nominal_files)
    
    all_variation_files = [x["path"] for x in file_dict["ttbar"]["PS_var"]["files"]]
    np.random.shuffle(all_variation_files)
    
    with open("carl_data_metadata.json", 'r') as f:
        data_metadata_dict = json.load(f)

    max_jet_size = 0
    max_electron_size = 0
    max_muon_size = 0
    for filename in all_nominal_files + all_variation_files:
        dataset = uproot.open(filename)["Events"].arrays(["nJet", "nElectron", "nMuon"])
        #max_jet_size = max([max(map(len, dataset[jet_features[0]])), max_jet_size])
        max_jet_size = max([ak.max(dataset["nJet"]), max_jet_size])
        #max_electron_size = max([max(map(len, dataset[electron_features[0]])), max_electron_size])
        max_electron_size = max([ak.max(dataset["nElectron"]), max_electron_size])
        #max_muon_size = max([max(map(len, dataset[muon_features[0]])), max_muon_size])
        max_muon_size = max([ak.max(dataset["nMuon"]), max_muon_size])
        
    data_metadata_dict["max_jet_size"] = max_jet_size
    data_metadata_dict["max_electron_size"] = max_electron_size
    data_metadata_dict["max_muon_size"] = max_muon_size
    
    with open("carl_data_metadata.json", 'w') as f:
        json.dump(data_metadata_dict, f)
        
    print("Done")