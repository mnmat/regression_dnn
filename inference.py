import torch
from utils.datasets import *
from models.models import *
from utils import validation
import pandas as pd
import sys
from utils.utils import *
import time
from utils.feature_mapping import FEATURES_RUN3_EB, FEATURES_RUN3_EE, FEATURES_RUN4_HLT
import argparse

# sys.path.append("/eos/home-m/mmatthew/SWAN_projects/BDT")

# from produce_plots import *
# from Validation.plots import *
# from Validation.hist_helpers import *
from utils.fit import fitCruijff, cruijff
from utils import training
from utils import datasets
from torch.utils.data import DataLoader

if __name__ == "__main__":

    # Load Settings
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path",type=str,default="saved_models/Run3",help="Model path")
    argparser.add_argument("--model_name",type=str,default="v1/",help="Model name")
    argparser.add_argument("--ideal_ic_file_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/Ideal",help="Path to input root files")
    argparser.add_argument("--proc_real_ic_file_path",type=str,default="/eos/home-m/mmatthew/SWAN_projects/egamma_regression/regression_dnn/data/Run3/Winter24/input_step2",help="Path to input root files")
    argparser.add_argument("--step", type=str,default="step1",help="Step 1 or step 2")
    argparser.add_argument("--isEB",action='store_true',help="If true, use EB model. If false, use EE model")
    argparser.add_argument("--outDir",type=str,default="/eos/home-m/mmatthew/www/Patatrack18/Winter24/",help="Output directory")
    args = argparser.parse_args()

    batchsize = 512
    device = torch.device("cpu")

    step = args.step
    if step == "step1":
        input_file_path = args.ideal_ic_file_path
        outDir = os.path.join(args.outDir,"Ideal/Validation")
    else:
        input_file_path = args.proc_real_ic_file_path
        outDir = os.path.join(args.outDir,"ProcReal/Validation")
    fnames = datasets.get_file_names(input_file_path)[-1]
    dataset = datasets.RegressionDatasetRun3([fnames],device=device,normalize="MinMax")
    dataloader = DataLoader(dataset,batch_size=batchsize,shuffle=True)


    model_path = os.path.join(args.model_path,args.step)
    isEB = args.isEB
    if not isEB:
        model_path = os.path.join(model_path,"EE")
    else:
        model_path = os.path.join(model_path,"EB")
    model_name = args.model_name

    outDir = os.path.join(outDir,model_name)
    create_dirs(outDir,True)

    if step == "step1": fix_mu = False
    else: fix_mu = True
    model = RegressionDNN_EGM(features=30, hidden_net=[512,512,512,512],fix_mu=fix_mu)
    device = torch.device("cpu")

    model_name = model_name+"/"+model_name.split("/")[0]+"_best"
    if model_path is not None:
        checkpoint = torch.load(os.path.join(model_path,model_name),map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    FEATURES_RUN3 = FEATURES_RUN3_EE
    ENERGIES = [0,100,200,300,400,500,600]


    d_hists = validation.create_histograms(ENERGIES,FEATURES_RUN3,"MinMax")
    with torch.no_grad():
        for batch,val_batch in enumerate(dataloader):
            start = time.time()
            inpt = val_batch["features"]
            orig_inpt = val_batch["orig_features"]
            tgt = val_batch["targets"]

            dscb  = model(inpt)

            gen_energy = val_batch["gen_energy"].detach().cpu()
            raw_energy = orig_inpt[:,FEATURES_RUN3["sc_rawEnergy"]].detach().cpu()
            if step == "step1":
                pred_energy = torch.exp(-dscb.mu.cpu().squeeze()) * raw_energy
            else:
                pred_energy = torch.exp(tgt.cpu()) * gen_energy
            
            d_hists = validation.fill_histograms(d_hists,inpt,tgt,raw_energy,pred_energy,gen_energy,dscb,ENERGIES,FEATURES_RUN3)
            print("batchnr. %s: "%batch,time.time()-start)

    validation.produce_plots(d_hists,ENERGIES,outDir)
    with open("histograms.pkl","wb") as f:
        pickle.dump(d_hists["histsParam_energies"],f)