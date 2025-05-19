import os
import json

from comet_ml import Experiment
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import uproot

import numpy as np
import pandas as pd
import awkward as ak
import time
import matplotlib.pyplot as plt
import math

from utils import training
from utils import datasets
from utils.datasets import *
from utils.training import *
from models import models

from utils.learning_rate_scheduler import CosineAnnealingWithLinearDecay


def count_parameters(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params


if __name__ == "__main__":

    with open("utils/comet_setup.json","r") as f:
        comet_setup = json.load(f)

    exp = Experiment(
            api_key=comet_setup["api_key"],
            project_name=comet_setup["project_name"]["run_3_step2"],
            workspace=comet_setup["workspace"],
            auto_output_logging = "simple",
            disabled= False
        )

    # Data Preprocessing

    path = "/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/input_step2"
    isEE = True

    batchsize = 512
    device = torch.device("cuda")

    fnames = datasets.get_file_names(path)
    
    if isEE:
        fnames = [fname for fname in fnames if "ee.pkl" in fname]
        features = FEATURES_RUN3_EE
    else:
        fnames = [fname for fname in fnames if "eb.pkl" in fname]
        features = FEATURES_RUN3_EB
    
    fnames = fnames[:-1] # Exclude the last file for testing
    split = math.ceil(len(fnames)*0.8)

    train_fnames = fnames[:split]
    test_fnames = fnames[split:]
    train_dataset = datasets.RegressionDatasetRun3(train_fnames,device=device,normalize="MinMax")
    test_dataset = datasets.RegressionDatasetRun3(test_fnames,device=device,normalize="MinMax")

    # Set Trainer 

    lr = 1e-4

    # Select Model
    alphaL_min = 0.5
    alphaR_min = 0.5
    model = models.RegressionDNN_EGM(features=30, hidden_net=[512,512,512,512],alphaL_min=alphaL_min,alphaR_min=alphaR_min,fix_mu=True)
    model.to(device)

    path = "saved_models/Run3/step2"
    if isEE:
        path = os.path.join(path,"EE")
    else:
        path = os.path.join(path,"EB")
    name = "v2"
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
    trainer = training.Trainer(model, train_dataset,test_dataset,batchsize, optimizer,path,name,isEE,scheduler=scheduler,shuffle=False)

    # Training
    epochs = 10
    trainer.full_train(epochs,exp)