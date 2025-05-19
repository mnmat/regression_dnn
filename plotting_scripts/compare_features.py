import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
from .feature_plots import *
from utils.utils import *
import sys


def compare_features():
    outdir = "/eos/home-m/mmatthew/www/Patatrack18/DoubleElectron_FlatPt-1To100-gun/Features/RootFiles/"

    # Samples
    training_samples = []
    test_samples = []

    path = "/eos/user/m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/GenSim/TICLv4_Mustache/electron"

    train,test = get_training_test_files(path,training_dir=".",test_dir=".")
    train = os.path.join(path,"HLTAnalyzerTree_IDEAL_Flat_train.root")
    test = os.path.join(path,"HLTAnalyzerTree_IDEAL_Flat_test.root")
    training_samples.append(train)
    test_samples.append(test)

    # path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/Spring23/DoubleElectron_FlatPt-1To100-gun/"
    # #path = "/eos/cms/store/group/dpg_hgcal/comm_hgcal/mmatthew/BDT/Samples/Spring24/DoubleElectron_FlatPt-1To100-gun/"
    # train,test = get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s5Reg_genMatched_HLT")
    # training_samples.append(train)
    # test_samples.append(test)
    names = ["Ticlv4"]

    plot_event_sizes(training_samples,test_samples,outdir,names)

    # Create Dataframes

    dfs = []
    key = "egRegDataHGCALHLTV1"

    outdir = outdir + "/" + key
    create_dirs(outdir,True)


    for file in test_samples:
        f = uproot.open(file)
        df = f[key].arrays(library="pd")
        #dfs.append(modify_tree(df))
        dfs.append(df)

    # # Plot Energies
    # for df,name in zip(dfs,names):
    #     plot_energies(df,outdir,name)
    #     scatter(df,outdir,name,ls=["raw","reg"])
    #     plot_fractions(df,outdir,name)

    # Plot features
    plot_feature_hists(dfs,names,outdir)

    # Plot correlation
    plot_feature_correlation(dfs,names,outdir)

if __name__ == "__main__":
    compare_features()
