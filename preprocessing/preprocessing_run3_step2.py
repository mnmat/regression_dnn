import numpy as np
import pandas as pd
import pickle
import torch
import sys
import os
from torch.utils.data import DataLoader
import argparse

from .preprocessing_lib import *
from utils import datasets
from models.models import *
from utils.feature_mapping import FEATURES_RUN3_EB, FEATURES_RUN3_EE, FEATURES_RUN4_HLT

if __name__ == "__main__":
    # Applies regression to Run 3 data with real IC 

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/saved_models/Run3/step1/",help="Model path")
    argparser.add_argument("--input_file_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/Real",help="Path to input root files")
    argparser.add_argument("--output_file_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/ProcReal",help="Output path")  
    argparser.add_argument("--isEB",action='store_true',help="If true, use EB model. If false, use EE model")
    argparser.add_argument("--model_name",type=str,default="v1/v1_best",help="Model name")
    args = argparser.parse_args()

    # Paths
    model_path = args.model_path
    model_name = args.model_name
    input_file_path = args.input_file_path
    output_file_path = args.output_file_path
    output_file_path = os.path.join(output_file_path,model_name.split("/")[0])
    if not os.path.exists(output_file_path):
        os.makedirs(output_file_path)

    # Load model
    isEB = args.isEB
    if not isEB:
        model_path = os.path.join(model_path,"EE")
    else:
        model_path = os.path.join(model_path,"EB")
    model = RegressionDNN_EGM(features=30, hidden_net=[512,512,512,512])
    device = torch.device("cuda")

    if model_path is not None:
        checkpoint = torch.load(os.path.join(model_path,model_name))
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Get file names
    fnames = datasets.get_file_names(input_file_path)
    if not isEB:
        fnames = [fname for fname in fnames if "ee.pkl" in fname]
        feature_map = FEATURES_RUN3_EE
    else:
        fnames = [fname for fname in fnames if "eb.pkl" in fname]
        feature_map = FEATURES_RUN3_EB

    # Apply regression and write new files
    for fname in fnames:
        reg_energy,dscb_params = apply_regression([fname],device,model,feature_map)
        write_step2_file(fname,reg_energy,output_file_path,feature_map)
