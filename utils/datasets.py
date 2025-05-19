import uproot
import os
import numpy as np
import torch
from torch.utils.data import Sampler, StackDataset, TensorDataset, Dataset, DataLoader, WeightedRandomSampler

FEATURES = {1:"nrHitsThreshold",
            2:"eta",
            3:"rawEnergy",
            4:"phiWidth",
            5:"rvar",
            6:"numberOfSubClusters",
            7:"clusterMaxDR"
           }

class RegressionDataset(Dataset):
    def __init__(self,root,fname,tree):
        f = uproot.open(os.path.join(root,fname))
        keys = f[tree].keys()
        
        input_keys = ["nrHitsThreshold","eta","rawEnergy","phiWidth","rvar","numberOfSubClusters","clusterMaxDR"]
        target_keys = ["eg_gen_energy"]
        input_arr = f[tree].arrays(input_keys,library="numpy")
        target_arr = f[tree].arrays(target_keys,library="numpy")
        self.features = torch.tensor(np.array(list(input_arr.values()))).T.float()
        self.targets = torch.tensor(np.array(list(target_arr.values()))).T.float()
        self.targets = self.targets/self.features[:,2].unsqueeze(-1)


    def __getitem__(self,index):
        return {
            "features": self.features[index],
            "targets": self.targets[index]
        }

    def __len__(self):
        return len(self.features)
