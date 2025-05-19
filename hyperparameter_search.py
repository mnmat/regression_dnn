import os
import comet_ml
import comet_ml.integration.ray
from comet_ml import Experiment
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from comet_ml import ExistingExperiment

import uproot

import numpy as np
import pandas as pd
import awkward as ak
import time
import matplotlib.pyplot as plt

from utils import training
from utils import datasets
from utils.datasets import *
from utils.training import *
from models import models

from utils.learning_rate_scheduler import CosineAnnealingWithLinearDecay

from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.comet import CometLoggerCallback

import time



def train_model(config,tune_log=True):
    from comet_ml.integration.ray import comet_worker_logger
    from ray.air import session

    with open("utils/comet_setup.json","r") as f:
        comet_setup = json.load(f)

    exp = Experiment(
            api_key=comet_setup["api_key"],
            project_name=comet_setup["project_name"]["hyperparameter_search"],
            workspace=comet_setup["workspace"],
            auto_output_logging = "simple",
            disabled= False
        )

    root = "/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/GenSim/TICLv4_Mustache/electron"
    root = "/eos/cms/store/group/phys_egamma/ec/prrout/EGM_regression_Ntuples_Winter24_CMSSW_13_3_X_18062024/IdealIC_WinterMC/1330/EgRegTree/AODSIM/DoubleElectron_FlatPT-1to500_13p6TeV/FlatPU0to120_133X_mcRun3_2024_realistic_v9_ECALIdealIC-v3_AODSIM_EgRegTree/240701_105939/0000"

    fname = "HLTAnalyzerTree_IDEAL_Flat_train.root"
    tree = "egRegDataHGCALHLTV1" 
    
    path = "saved_models"
    name = "v7"
    path = os.path.join(path,name)
    
    epochs = 5
    
    alphaL_min = config["alphaL_min"]
    alphaR_min = config["alphaR_min"]
    learning_rate = config["learning_rate"]
    batchsize = config["batchsize"]
    hidden_layers = config["hidden_layers"]
    enable_gpu = config["enable_gpu"]
    tag = config["tag"]


    exp.add_tags([tag])

    if enable_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    dataset = RegressionDataset(root,fname,tree,device=device,normalize="MinMax")
    dataset = RegressionDatasetRun3(root)
    
    exp.log_parameters({
        "alphaL_min": alphaL_min,
        "alphaR_min": alphaR_min,
        "learning_rate": learning_rate,
        "batchsize": batchsize,
        "hidden_layers": hidden_layers,
        "epochs": epochs,
        "device": str(device)
    })
    
    # Select Model
    model = models.RegressionDNN_EGM(features=7, hidden_net=hidden_layers,alphaL_min=alphaL_min, alphaR_min=alphaR_min)
    model.to(device)

    # Select optimiser
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate)

    # Initialize LR Scheduler
    scheduler = CosineAnnealingWithLinearDecay(optimizer,15,lr_min=1e-6,max_lr_start=1e-4,max_lr_end=1e-6,num_cycles=4)

    # Set Training and test datasets
    train_dataset, test_dataset = random_split(dataset,[0.8,0.2])

    # Initialize Trainer
    trainer = training.Trainer(model, train_dataset, test_dataset,batchsize, optimizer,path,name,scheduler=scheduler)

    # Training
    start = time.time()
    trainer.full_train(epochs,exp,tune_log=tune_log)
    #exp.log_metric("training_time", time.time() - start)
#     for i in range(30):
#         tune.report({"validation_loss":i})

    #tune.report({"validation loss":trainer.metrics["validation_loss"]})



def setup_experiment(comet_token,project_name,workspace):
    exp = Experiment(
        api_key=comet_token,
        project_name=project_name,
        workspace=workspace,
        auto_output_logging = "simple",
        disabled= False
        )
    return exp


if __name__ == "__main__":

    scheduler = ASHAScheduler(metric="validation_loss", mode="min")
    trial_space = {"alphaL_min":tune.uniform(0.1,1.5),
            "alphaR_min":tune.uniform(0.1,1.5),
            "learning_rate":tune.loguniform(1e-5,1e-2),
            "batchsize":tune.choice([128,256,512]),
            "hidden_layers":tune.grid_search([[64,64,64],[128,128,128],[256,256,256],[512,512,512]]),
            "enable_gpu": True,
            "tag":"debug_search_manual_metrics_only"}


    run_config=tune.RunConfig(
            callbacks=[
                CometLoggerCallback(
                    api_key=comet_token, project_name=project_name, tags=["debug_search"]
                )
            ],
        )

    tune_config = tune.TuneConfig(num_samples = 20,
                                scheduler = scheduler)
    #                              metric="validation_loss",
    #                              mode="min")

    trainable_with_resources = tune.with_resources(train_model, {"gpu": 1})

    tuner = tune.Tuner(trainable_with_resources 
                    ,param_space=trial_space
                    ,tune_config = tune_config)
                    #,run_config = run_config)

    results = tuner.fit()

