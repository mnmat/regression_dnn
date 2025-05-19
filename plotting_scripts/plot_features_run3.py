import uproot
import os
import pickle
import hist
import mplhep as hep
import argparse

import mplhep as hep
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys

from utils.feature_mapping import FEATURES_RUN3_EB, FEATURES_RUN3_EE, FEATURES_RUN4_HLT
from utils.utils import get_variables, get_mask, get_gen_energy, create_dirs
from utils.datasets import get_file_names


def create_min_max(feature_map):
    min_max = {}
    names = feature_map.keys()
    for name in names:
        min_max[name] = {"min":999,
                        "max":-999}
    min_max["cp_energy"] = {"min":999,
                        "max":-999}
    min_max["tgt"] = {"min":999,
                     "max":-999}
    return min_max

def fill_min_max(feature_map,gen,features,min_max):
    names = feature_map.keys()
    for name in names:
        min_ = np.nanmin(features[:,feature_map[name]])
        max_ = np.nanmax(features[:,feature_map[name]])
        if min_ < min_max[name]["min"]:
            min_max[name]["min"] = min_
        if max_ > min_max[name]["max"]:
            min_max[name]["max"] = max_
    
    min_ = gen.min()
    max_ = gen.max()
    if min_ < min_max["cp_energy"]["min"]:
        min_max["cp_energy"]["min"] = min_ 
    if max_ > min_max["cp_energy"]["max"]:
        min_max["cp_energy"]["max"] = max_
        
    
    tgt = np.log(features[:,feature_map["sc_rawEnergy"]]/gen)
    
    min_ = np.nanmin(tgt[np.isfinite(tgt)])
    max_ = np.nanmax(tgt[np.isfinite(tgt)])
    if min_ < min_max["tgt"]["min"]:
        min_max["tgt"]["min"] = min_ 
    if max_ > min_max["tgt"]["max"]:
        min_max["tgt"]["max"] = max_
    
    return min_max
    

def get_min_max(fnames,feature_map_ee, feature_map_eb,mode,load=True):
    if load:
        try:
            with open('min_max_ee_%s.p'%mode, 'rb') as fp:
                min_max_ee = pickle.load(fp)
            with open('min_max_eb_%s.p'%mode, 'rb') as fp:
                min_max_eb = pickle.load(fp)
            
            return min_max_ee, min_max_eb
        except:
            print("Load from storage file failed. Continue to read from root files")
        
    min_max_ee = create_min_max(feature_map_ee)
    min_max_eb = create_min_max(feature_map_eb)

    for fname in fnames:

        f = uproot.open(fname)
        tree = f["egRegTree"]

        features_ee, features_eb = get_variables(tree)
        gen_ee, gen_eb = get_gen_energy(tree)

        min_max_ee = fill_min_max(feature_map_ee,gen_ee,features_ee,min_max_ee)
        min_max_eb = fill_min_max(feature_map_eb,gen_eb,features_eb,min_max_eb)

    with open('min_max_ee_%s.p'%mode, 'wb') as fp:
        pickle.dump(min_max_ee, fp, protocol=pickle.HIGHEST_PROTOCOL)

    with open('min_max_eb_%s.p'%mode, 'wb') as fp:
        pickle.dump(min_max_eb, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return min_max_ee, min_max_eb

def create_histograms(features,min_max,truth="cp_energy"):

    bins=50
    hist2D = {}
    names = features
    for name in names:
        if "nergy" in name and not "/" in name and truth == "cp_energy":
            if min_max[name]["min"] < min_max[truth]["min"]:
                ax_min = min_max[name]["min"]*1.01
            else:
                ax_min =  min_max["cp_energy"]["min"]*1.01
            if min_max[name]["max"] > min_max[truth]["max"]:
                ax_max = min_max[name]["max"]*1.01
            else:
                ax_max =  min_max["cp_energy"]["max"]*1.01
            hist2D[name] = hist.Hist(
                hist.axis.Regular(bins,ax_min,ax_max,name=name,overflow=True),
                hist.axis.Regular(bins,ax_min,ax_max,name=truth,overflow=True)
            )

        else:
            hist2D[name] = hist.Hist(
                hist.axis.Regular(bins,min_max[name]["min"]*1.01,min_max[name]["max"]*1.01,name=name,overflow=True),
                hist.axis.Regular(bins,min_max[truth]["min"]*1.01,min_max[truth]["max"]*1.01,name=truth,overflow=True)
            )
    return hist2D


def fill_histograms(features,gen,feature_map,hist2D,truth="cp_energy"):
    target_arr = np.log(features[:,feature_map["sc_rawEnergy"]]/gen)
    mask = np.where(~np.isinf(target_arr))
    
    features = features[mask]
    gen = gen[mask]
    tgt = target_arr[mask]
    
    names = feature_map.keys()
    for name in names:
        if truth == "cp_energy":
            hist2D[name].fill(features[:,feature_map[name]],gen) 
        else:
            hist2D[name].fill(features[:,feature_map[name]],tgt) 
    return hist2D


def write_histograms(hist2D,feature_map,root,truth="cp_energy",save=True):

    hep.style.use("CMS")

    if truth == "cp_energy":
        outdir = os.path.join(root,"Features")
    else:
        outdir = os.path.join(root,"FeaturesTgt")
    create_dirs(outdir,True)
    
    names = feature_map.keys()
    for name in names:
        fig,ax = plt.subplots()
        hist2D[name].project(name).plot(ax=ax)
        if "/" in name:
            name = name.split("/")[0]+"_o_"+name.split("/")[1]
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))
            
            
    # TODO: Find better solution to plot the tgt variables
    if truth == "cp_energy":
        fig,ax = plt.subplots()
        hist2D[name].project("cp_energy").plot(ax=ax)
        name = "cp_energy"
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))
    elif truth == "tgt":
        fig,ax = plt.subplots()
        hist2D[name].project("tgt").plot(ax=ax)
        name = "tgt"
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))
        
            

    if truth == "cp_energy":
        outdir = os.path.join(root,"Correlation")
    else:
        outdir = os.path.join(root,"CorrelationTgt2")
    create_dirs(outdir,True)
    save = True

    for name in names:
        fig,ax = plt.subplots()
        hist2D[name].plot(cmap="plasma",cmin=1,ax=ax)
        if "/" in name:
            name = name.split("/")[0]+"_o_"+name.split("/")[1]
        if save:
            fig.savefig(os.path.join(outdir,"%s.png"%name))
            fig.savefig(os.path.join(outdir,"%s.pdf"%name))



if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='Plot features for Run3 samples')
    argparser.add_argument('--mode', type=str, default="Ideal",
                        help='Ideal or Real IC')
    argparser.add_argument('--outDir', type=str, default="/eos/home-m/mmatthew/www/Patatrack18/Winter24/",
                        help='Output directory')
    args = argparser.parse_args()

    hep.style.use("CMS")

    # Get files
    mode = args.mode
    if mode == "Ideal":
        path = "/eos/cms/store/group/phys_egamma/ec/prrout/EGM_regression_Ntuples_Winter24_CMSSW_13_3_X_18062024/IdealIC_WinterMC/1330/EgRegTree/AODSIM/DoubleElectron_FlatPT-1to500_13p6TeV/FlatPU0to120_133X_mcRun3_2024_realistic_v9_ECALIdealIC-v3_AODSIM_EgRegTree/240701_105939/0000"
    else:
        path = "/eos/cms/store/group/phys_egamma/ec/prrout/EGM_regression_Ntuples_Winter24_CMSSW_13_3_X_18062024/RealIC_Winter24MC/1330/EgRegTree/AODSIM/DoubleElectron_FlatPT-1to500_13p6TeV/FlatPU0to120_133X_mcRun3_2024_realistic_v9-v2_AODSIM_EgRegTree/240619_154812/0000"
    fnames = get_file_names(path)

    # Set output directory
    outDir = args.outDir
    outDir = outDir + "/Features_Run3_%s/"%mode

    # Get min and max for each feature
    min_max_ee, min_max_eb = get_min_max(fnames,FEATURES_RUN3_EE,FEATURES_RUN3_EB,mode,load=True)

    # Plot all features and correlations (gen_energy)
    hist2D_ee = create_histograms(FEATURES_RUN3_EE,min_max_ee)
    hist2D_eb = create_histograms(FEATURES_RUN3_EB,min_max_eb)
    for fname in fnames:
                
        f = uproot.open(fname)
        tree = f["egRegTree"]

        features_ee, features_eb = get_variables(tree)
        gen_ee, gen_eb = get_gen_energy(tree)

        hist2D_ee = fill_histograms(features_ee,gen_ee,FEATURES_RUN3_EE,hist2D_ee)
        hist2D_eb = fill_histograms(features_eb,gen_eb,FEATURES_RUN3_EB,hist2D_eb)

    write_histograms(hist2D_ee,FEATURES_RUN3_EE,os.path.join(outDir,"EE"),save=True)
    write_histograms(hist2D_eb,FEATURES_RUN3_EB,os.path.join(outDir,"EB"),save=True)

    # Plot all features and correlations (tgt)
    hist2D_ee = create_histograms(FEATURES_RUN3_EE,min_max_ee,truth="tgt")
    hist2D_eb = create_histograms(FEATURES_RUN3_EB,min_max_eb,truth="tgt")

    for fname in fnames:
                
        f = uproot.open(fname)
        tree = f["egRegTree"]

        features_ee, features_eb = get_variables(tree)
        gen_ee, gen_eb = get_gen_energy(tree)

        hist2D_ee = fill_histograms(features_ee,gen_ee,FEATURES_RUN3_EE,hist2D_ee,truth="tgt")
        hist2D_eb = fill_histograms(features_eb,gen_eb,FEATURES_RUN3_EB,hist2D_eb,truth="tgt")
    write_histograms(hist2D_ee,FEATURES_RUN3_EE,os.path.join(outDir,"EE"),truth="tgt",save=True)
    write_histograms(hist2D_eb,FEATURES_RUN3_EB,os.path.join(outDir,"EB"),truth="tgt",save=True)