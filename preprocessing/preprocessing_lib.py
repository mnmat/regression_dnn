import numpy as np
import pandas as pd
import pickle
import torch
import sys
import os
from copy import deepcopy
from torch.utils.data import DataLoader
import uproot
# sys.path.append("/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn")

from utils import datasets
from models.models import *
from utils.utils import *

def apply_regression(fname,device,model,features):
    
    batches = 512

    val_dataset = datasets.RegressionDatasetRun3(fname,device=device,normalize="MinMax")
    dataloader = DataLoader(val_dataset,batch_size=batches,shuffle=False)

    reg_energy = []
    params = {
        "mu":[],
        "sigma":[],
        "alphaL":[],
        "alphaR":[],
        "etaL":[],
        "etaR":[]
    }
    
    for batch,val_batch in enumerate(dataloader):
        print(batch)
        inpt = val_batch["features"].to(device)
        tgt = val_batch["targets"].to(device)
        dscb  = model(inpt)

        output = torch.exp(-dscb.mu).detach().cpu()
        inpt = val_batch["orig_features"].cpu()
        tgt = torch.exp(-tgt.cpu())
        cp_energy = tgt.T*inpt[:,features["sc_rawEnergy"]].flatten()

        reg_energy.append((output.T*inpt[:,features["sc_rawEnergy"]])[0])
        params["mu"].append(dscb.mu.flatten().detach().cpu().numpy())
        params["sigma"].append(dscb.sigma.flatten().detach().cpu().numpy())
        params["alphaL"].append(dscb.alphaL.flatten().detach().cpu().numpy())
        params["alphaR"].append(dscb.alphaR.flatten().detach().cpu().numpy())
        params["etaL"].append(dscb.etaL.flatten().detach().cpu().numpy())
        params["etaR"].append(dscb.etaR.flatten().detach().cpu().numpy())
        
        
    reg_energy = np.concatenate(reg_energy)
    for key in params.keys():
        params[key] = np.concatenate(params[key])
    return reg_energy, params

def write_step2_file(fname,reg_energy,output_file_path,features):
    with open(fname,"rb") as infile:

        x = pickle.load(infile)
        inpt_features = x["features"]
        raw_energy = deepcopy(inpt_features[:,features["sc_rawEnergy"]])
        gen_energy = x["gen_energy"]

        #inpt_features[:,features["sc_rawEnergy"]]=reg_energy

        data={"features": inpt_features,
          "gen_energy":gen_energy,
          "reco_energy":reg_energy,
          "raw_energy":raw_energy,
          "target":np.log(reg_energy/gen_energy)}

        name = fname.split("/")[-1]
        name = "reg_step1_" + name
        out_name = os.path.join(output_file_path,name)  
        with open(out_name,"wb") as outfile:
            pickle.dump(data,outfile)
            
def write_regression_to_file(fname,reg_energy,params,output_file_path,features):
    with open(fname,"rb") as infile:

        x = pickle.load(infile)
        inpt_features = x["features"].copy()
        raw_energy = deepcopy(inpt_features[:,features["sc_rawEnergy"]])
        gen_energy = x["gen_energy"]

        data={"gen_energy":gen_energy,
          "reco_energy":reg_energy,
          "raw_energy":raw_energy,
          "target":np.log(raw_energy/gen_energy),
          "target_reg":np.log(reg_energy/gen_energy),
          "params":params
        }

        name = fname.split("/")[-1]
        name = "reg_step1_" + name
        out_name = os.path.join(output_file_path,name)  
        with open(out_name,"wb") as outfile:
            pickle.dump(data,outfile)


def write_to_file(features,gen,fname,root,isEE):
    features = np.array(features)
    gen = np.array(gen)
    
    target = np.log(features[:,1]/gen)
    
    mask = np.where(~np.isinf(target))
    
    target = target[mask]
    features = features[mask]
    gen = gen[mask]

    name = fname.split("/")[-1].split(".root")[0]
    if isEE:
        name = name + "_ee"
    else:
        name = name + "_eb"
        
    outfile = os.path.join(root,name)
    
    data = {"features":features,
       "target":target,
        "gen_energy":gen}
    
    with open(outfile+".pkl","wb") as f:
        pickle.dump(data,f)
    
def preprocess_file(root,fname):

    f = uproot.open(fname)
    tree = f["egRegTree"]
    features_ee, features_eb = get_variables(tree)
    gen_ee, gen_eb = get_gen_energy(tree)

    write_to_file(features_ee,gen_ee,fname,root,True)
    write_to_file(features_ee,gen_ee,fname,root,False)