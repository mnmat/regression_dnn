import torch
from torch.utils.data import Dataset, DataLoader,random_split
from torch import nn
from tqdm.auto import tqdm
from utils import validation
import os
import matplotlib.pyplot as plt
from ray import tune
import time
import pdb
import sys

from utils.feature_mapping import FEATURES_RUN3_EB, FEATURES_RUN3_EE, FEATURES_RUN4_HLT

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Trainer:
    def __init__(self, model:nn.Module, train_dataset:Dataset,test_dataset,batchsize, optimizer,outdir,name,isEB=False,normalize="MinMax",shuffle=True, scheduler=None, device="cpu") -> None:
        self.model = model
        self.train_dataloader = DataLoader(train_dataset,batch_size=batchsize,shuffle=shuffle)
        self.validation_dataloader = DataLoader(test_dataset,batch_size=batchsize,shuffle=shuffle)
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.scheduler = scheduler

        self.losses_per_epoch = []
        self.losses_per_batch = []

        self.name = name
        self.outdir = outdir
        self.normalize = normalize
        create_dir(outdir)


        if "Run3" in str(type(train_dataset)):
            if not isEB:
                self.feature_map = FEATURES_RUN3_EE
            else:
                self.feature_map = FEATURES_RUN3_EB
        else:
            self.feature_map = FEATURES_RUN4_HLT


        # torch.save(model.state_dict(), os.path.join(outdir,"state.txt"))

        self.metrics = {"training_loss":[],
                       "validation_loss":[]}

    def train_loop(self,epoch,comet=None):
        
        self.model.train()
        training_loss = 0
        for batch, data_batch in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            #data_batch_device = {key: val.to(self.device) for key, val in data_batch.items()}
            inpt = data_batch["features"] #.to(self.device)
            tgt = data_batch["targets"] #.to(self.device)
            logprob = self.model.log_prob(tgt,inpt)
            loss = -logprob.mean()

            training_loss += loss.item()
            
            # Backpropagation
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        
        if self.scheduler: self.scheduler.step()

        training_loss/=(batch+1)
        
        if comet:
            comet.log_metric("Training Loss",training_loss,step=epoch)

        self.metrics["training_loss"].append(training_loss)

    def validation_loop(self,epoch,comet=None):
        self.model.eval()
        validation_loss = 0

        # Visualization

        hists = validation.createHists()
        hist2D = validation.createHists2D()
        histsParam = validation.createHistsParameter()
        histsParam2D = validation.createHistsParameter2D(self.feature_map,self.normalize)
        
        energies = [0,100,200,300,400,500,600]
        energy_labels = ["%s_%s"%(energies[i],energies[i+1]) for i in range(len(energies)-1)]
        hists_energies = []
        histsParam_energies = []
        histsParam2D_energies = []
        for i in range(len(energies)-1):
            hists_energies.append(validation.createHists())
            histsParam_energies.append(validation.createHistsParameter())
            histsParam2D_energies.append(validation.createHistsParameter2D(self.feature_map,self.normalize))
        
        # plotsToSave = [plotFullEnergies, plotRatioOverCP]
        # plotsToSave2D = [plotTrueVPred]

        
        for batch, data_batch in tqdm(enumerate(self.validation_dataloader), total=len(self.validation_dataloader)):
            #data_batch_device = {key: val.to(self.device) for key, val in data_batch.items()}
            inpt = data_batch["features"] #.to(self.device)
            tgt = data_batch["targets"] #.to(self.device)
            logprob = self.model.log_prob(tgt,inpt)
            loss = -logprob.mean()
            validation_loss += loss.item()

            dscb  = self.model(inpt)

            output = torch.exp(-dscb.mu).detach().cpu() # E_pred/E_reco

            inpt_orig = data_batch["orig_features"].cpu()
            tgt = torch.exp(-tgt.cpu()) # E_true/E_reco

            inpt = inpt.cpu()
            output = output.cpu()

            gen_energy = data_batch["gen_energy"].detach().cpu()
            raw_energy = inpt_orig[:,self.feature_map["sc_rawEnergy"]].flatten()
            reg_energy = output.squeeze()*raw_energy
            
            hists = validation.writeHistograms(raw_energy,reg_energy,gen_energy,hists)
            hists2D = validation.writeHistograms2D(reg_energy,gen_energy,hist2D)
        
            params = [torch.exp(dscb.mu.detach().cpu().squeeze()),
                      dscb.sigma.detach().cpu().squeeze(),
                      dscb.alphaL.detach().cpu().squeeze(),
                      dscb.alphaR.detach().cpu().squeeze(),
                      dscb.etaL.detach().cpu().squeeze(),
                      dscb.etaR.detach().cpu().squeeze()]
            
            histsParam = validation.writeHistogramsParameter(params,histsParam)
            histsParam2D = validation.writeHistogramsParameter2D(params,histsParam2D,tgt.flatten(),inpt,output.flatten(),self.feature_map)
        
            for i in range(len(energies)-1):

                # TODO:
                # For the run3 data for some reason I need [0]. Is this the same for the run4 data, needs [1]?
                mask = torch.where((gen_energy >= energies[i]) & (gen_energy<energies[i+1]))[0] 
                hists_energies[i] = validation.writeHistograms(raw_energy[mask],reg_energy[mask],gen_energy[mask],hists_energies[i])
        
                params = [torch.exp(dscb.mu[mask].detach().cpu().squeeze()),
                          dscb.sigma[mask].detach().cpu().squeeze(),
                          dscb.alphaL[mask].detach().cpu().squeeze(),
                          dscb.alphaR[mask].detach().cpu().squeeze(),
                          dscb.etaL[mask].detach().cpu().squeeze(),
                          dscb.etaR[mask].detach().cpu().squeeze()]
                
                histsParam_energies[i] = validation.writeHistogramsParameter(params,histsParam_energies[i])
                histsParam2D_energies[i] = validation.writeHistogramsParameter2D(params,histsParam2D_energies[i],tgt[mask].flatten(),inpt[mask],output[mask].flatten(), self.feature_map)


        validation_loss/=(batch+1)

        self.metrics["validation_loss"].append(validation_loss)
        
        if comet:
            comet.log_metric("Validation Loss",validation_loss,step=epoch)
            
            # for plotFct in plotsToSave:
            #     fig = plotFct(hists)
            #     comet.log_figure(plotFct.__name__,fig,step=epoch)

            # for plotFct in plotsToSave2D:
            #     fig = plotFct(hists2D)
            #     comet.log_figure(plotFct.__name__,fig,step=epoch)
            
            # 1D hists
            plotsToSave = [validation.plotFullEnergies, validation.plotRatioOverCP,validation.plotTargetPred]
            validation.plot_histograms(hists,plotsToSave,epoch=epoch,comet=comet)
            for i in range(len(energies)-1):
                validation.plot_histograms(hists_energies[i],plotsToSave,epoch=epoch,name=str(energies[i]),comet=comet)
            
            # 2D hists
            plotsToSave = [validation.plotTrueVPred]
            validation.plot_histograms(hist2D,plotsToSave,epoch=epoch,comet=comet)
            
            # Parameters
            validation.plot_parameters(histsParam,epoch=epoch,comet=comet)
            # for i in range(len(energies)-1):
            #     validation.plot_parameters(histsParam_energies[i],epoch=epoch,comet=comet,name=str(energies[i]))
            
            validation.plot_parameters2D(histsParam2D,epoch=epoch,comet=comet)
            # for i in range(len(energies)-1):
            #     validation.plot_parameters2D(histsParam2D_energies[i],epoch=epoch,comet=comet,name=str(energies[i]))

        return validation_loss

    def full_train(self, nepochs,comet=None,tune_log=None):
        best_loss = 999
        for epoch in range(nepochs):
            start = time.time()
            print("########## Epoch " + str(epoch))
            self.train_loop(epoch,comet)
            validation_loss = self.validation_loop(epoch,comet)
            end = time.time()

            if tune_log is not None:
                tune.report({"validation_loss":validation_loss})

            if comet is not None:
                comet.log_metric("epoch_time",end-start,epoch)

            if self.scheduler is not None:
                self.scheduler.step(validation_loss)    

            # save best models
            if validation_loss < best_loss:
                self.save(epoch,"best")

        self.save(epoch,"last")
        
    
    def save(self,epoch,label, **kwargs):
        path = os.path.join(self.outdir,self.name+"_"+label)
        
        torch.save({
            'epoch':epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            **kwargs
            }, path)