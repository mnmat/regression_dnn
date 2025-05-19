import numpy as np
import pandas as pd
import os
import sys
import awkward as ak
import argparse

from .preprocessing_lib import *
from utils.datasets import *
from utils.feature_mapping import FEATURES_RUN3_EB, FEATURES_RUN3_EE, FEATURES_RUN4_HLT
from utils.utils import *


if __name__ == "__main__":
    # Create flat pkl files from Run3 Winter 24 ntuples for input to pytorch training

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--output_file_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/",help="Path to input root files")
    args = argparser.parse_args()
    root = args.output_file_path

    pathDir = {
        "IdealIC": "/eos/cms/store/group/phys_egamma/ec/prrout/EGM_regression_Ntuples_Winter24_CMSSW_13_3_X_18062024/IdealIC_WinterMC/1330/EgRegTree/AODSIM/DoubleElectron_FlatPT-1to500_13p6TeV/FlatPU0to120_133X_mcRun3_2024_realistic_v9_ECALIdealIC-v3_AODSIM_EgRegTree/240701_105939/0000",
        "RealIC": "/eos/cms/store/group/phys_egamma/ec/prrout/EGM_regression_Ntuples_Winter24_CMSSW_13_3_X_18062024/RealIC_Winter24MC/1330/EgRegTree/AODSIM/DoubleElectron_FlatPT-1to500_13p6TeV/FlatPU0to120_133X_mcRun3_2024_realistic_v9-v2_AODSIM_EgRegTree/240619_154812/0000"
    }

    for key in pathDir.keys():
        outDir = os.path.join(root,key)
        if not os.path.exists(outDir):
            os.makedirs(outDir)
        
        fnames = get_file_names(pathDir[key])
        for fname in fnames:
            preprocess_file(outDir,fname)