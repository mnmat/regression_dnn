import os
from comet_ml import Experiment
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import json

import uproot

import numpy as np
import pandas as pd
import awkward as ak
import time
import matplotlib.pyplot as plt
import math
import argparse

from utils import training
from utils import datasets
from utils.datasets import *
from utils.training import *
from models import models

from utils.learning_rate_scheduler import CosineAnnealingWithLinearDecay
from utils.feature_mapping import FEATURES_RUN3_EB, FEATURES_RUN3_EE, FEATURES_RUN4_HLT


if __name__ == "__main__":

    # Load Settings
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path",type=str,default="saved_models/Run3",help="Model path")
    argparser.add_argument("--model_name",type=str,default="v1/",help="Model name")
    argparser.add_argument("--ideal_ic_file_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/Ideal",help="Path to input root files")
    argparser.add_argument("--proc_real_ic_file_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/input_step2",help="Path to input root files")
    argparser.add_argument("--step", type=str,default="step1",help="Step 1 or step 2")
    argparser.add_argument("--isEB",action='store_true',help="If true, use EB model. If false, use EE model")
    args = argparser.parse_args()

    step = args.step
    with open("utils/comet_setup.json","r") as f:
        comet_setup = json.load(f)

    exp = Experiment(
            api_key=comet_setup["api_key"],
            project_name=comet_setup["project_name"]["run3_%s"%(step)],
            workspace=comet_setup["workspace"],
            auto_output_logging = "simple",
            disabled= False
        )


    # Get Dataset
    if step == "step1":
        input_file_path = args.ideal_ic_file_path
    else:
        input_file_path = args.proc_real_ic_file_path
    fnames = datasets.get_file_names(input_file_path)

    isEB = args.isEB
    if not isEB:
        fnames = [fname for fname in fnames if "ee.pkl" in fname]
        features = FEATURES_RUN3_EE
    else:
        fnames = [fname for fname in fnames if "eb.pkl" in fname]
        features = FEATURES_RUN3_EB

    fnames = fnames[:-1] # Exclude the last file for testing
    split = math.ceil(len(fnames)*0.8)
    train_fnames = fnames[:split]
    test_fnames = fnames[split:]

    batchsize = 512
    device = torch.device("cuda")

    train_dataset = datasets.RegressionDatasetRun3(train_fnames,device=device,normalize="MinMax")
    test_dataset = datasets.RegressionDatasetRun3(test_fnames,device=device,normalize="MinMax")


    # Set Trainer 

    lr = 1e-4

    # Select Model
    alphaL_min = 0.5
    alphaR_min = 0.5
    if step == "step1": fix_mu = False
    else: fix_mu = True
    model = models.RegressionDNN_EGM(features=30, hidden_net=[512,512,512,512],alphaL_min=alphaL_min,alphaR_min=alphaR_min,fix_mu=fix_mu)
    model.to(device)

    path = args.model_path
    path = "%s/%s"%(path,step)
    if not isEB:
        path = os.path.join(path,"EE")
    else:
        path = os.path.join(path,"EB")
    name = args.model_name
    path = os.path.join(path,name)

    # outdir = "saved_models/v2"
    # create_dir(outdir)
    # fname = "v2"
    # trainer.save(os.path.join(outdir,fname))

    # # Select Loss function
    # loss_fn = nn.MSELoss()

    # Select optimiser
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)

    # Initialize LR Scheduler
    scheduler = CosineAnnealingWithLinearDecay(optimizer,15,lr_min=1e-6,max_lr_start=1e-4,max_lr_end=1e-6,num_cycles=4)

    # Initialize Trainer
    trainer = training.Trainer(model, train_dataset,test_dataset,batchsize, optimizer,path,name,isEB,scheduler=scheduler,shuffle=False)

    # Training
    epochs = 10
    trainer.full_train(epochs,exp)
