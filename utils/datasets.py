import uproot
import os
import numpy as np
import torch
from torch.utils.data import Sampler, StackDataset, TensorDataset, Dataset, DataLoader, WeightedRandomSampler
import pickle


FEATURES = {"nrHitsThreshold":0,
            "eta":1,
            "sc_rawEnergy":2,
            "phiWidth":3,
            "rvar":4,
            "numberOfSubClusters":5,
            "clusterMaxDR":6
           }

class RegressionDataset(Dataset):
    def __init__(self,root,fname,tree,device="cpu",normalize=None):
        f = uproot.open(os.path.join(root,fname))
        keys = f[tree].keys()
        
        input_keys = ["nrHitsThreshold","eta","rawEnergy","phiWidth","rvar","numberOfSubClusters","clusterMaxDR"]
        target_keys = ["eg_gen_energy"]
        input_arr = f[tree].arrays(input_keys,library="numpy")
        #input_arr[input_keys.index("clusterMaxDR")] = input_arr[input_keys.index("clusterMaxDR")]
        target_arr = f[tree].arrays(target_keys,library="numpy")

        # TODO: Check for NaN

        self.features = torch.tensor(np.array(list(input_arr.values()))).T.float().to(device)
        self.orig_features = self.features
        self.targets = torch.tensor(np.array(list(target_arr.values()))).T.float().to(device).squeeze()
        # log(E_raw/E_true) as per AN2020 paper
        self.targets = torch.log(self.features[:,2]/self.targets)

        if normalize:
            if normalize == "Standardization":
                self.features = (self.features - self.features.mean(dim=0)) / self.features.std(dim=0)
            elif normalize == "MinMax":
                self.features = (self.features - self.features.min(dim=0).values) / (self.features.max(dim=0).values - self.features.min(dim=0).values)
            else:
                print("Normalization not implemented! Skip normalization")
            


    def __getitem__(self,index):
        return {
            "features": self.features[index],
            "targets": self.targets[index],
            "orig_features": self.orig_features[index]}

    def __len__(self):
        return len(self.features)


def get_file_names(path):
    fs = []
    for root, dirs, files in os.walk(path):
        for name in files:
            fs.append(os.path.join(root,name))
    return fs

class RegressionDatasetRun3(Dataset):
    def __init__(self,fnames,device="cpu",normalize=None):
        
        self.device = device
        self.fnames = fnames
        lengths = []
        for fname in self.fnames:
            with open(fname,"rb") as f:
                x = pickle.load(f)
                length = len(x["features"])
                lengths.append(length)
            
        self.cumsum = np.array(lengths).cumsum()
        self.file_index = 0
        self.normalize = normalize
        
        self.load_file(self.fnames[self.file_index])

    def load_file(self,fname):
        with open(fname,"rb") as f:
            x = pickle.load(f) 


            self.features = torch.tensor(x["features"]).float().to(self.device)
            self.orig_features = self.features
            self.targets = torch.tensor(x["target"]).float().to(self.device)
            self.gen_energy = torch.tensor(x["gen_energy"]).float().to(self.device)

            if self.normalize:
                if self.normalize == "Standardization":
                    self.features = (self.features - self.features.mean(dim=0)) / self.features.std(dim=0)
                elif self.normalize == "MinMax":
                    self.features = (self.features - self.features.min(dim=0).values) / (self.features.max(dim=0).values - self.features.min(dim=0).values)
                else:
                    print("Normalization not implemented! Skip normalization")

    def __getitem__(self,index):  
        
        if self.file_index != np.where(index<self.cumsum)[0][0]:
            self.file_index = np.where(index<self.cumsum)[0][0]
            self.load_file(self.fnames[self.file_index])
            
        if self.file_index-1 == -1:
            idx = index
        else:
            idx = index - self.cumsum[self.file_index-1]
        
        
        return {
            "features": self.features[idx],
            "targets": self.targets[idx],
            "orig_features": self.orig_features[idx],
            "gen_energy": self.gen_energy[idx]}

    def __len__(self):
        return self.cumsum[-1]