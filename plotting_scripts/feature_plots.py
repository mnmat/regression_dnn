import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches


def plot_event_sizes(training_samples,test_samples,outdir,names,keys = ["egRegDataEcalHLTV1","egRegDataHGCALHLTV1"],labels = ["Ecal","HGCAL"]):
    for train,test,name in zip(training_samples,test_samples,names):
        y_train = []
        y_test = []
    
        f_train = uproot.open(train)
        for key in keys:
            y_train.append(f_train[key].num_entries)

     
        f_test = uproot.open(test)
        for key in keys:
            y_test.append(f_test[key].num_entries)

        x_train = np.linspace(0,len(y_train)-1,len(y_train))
        x_test = np.linspace(0,len(y_test)-1,len(y_test))
        width=0.25
        offset = width/2

        fig, ax = plt.subplots()
        r1=ax.bar(x_train-offset,y_train, width=width,label="Training Sample: %i"%np.array(y_train).sum())
        ax.bar_label(r1,padding=-15,color="white")
        r2 = ax.bar(x_test+offset,y_test, width=width,label="Test Sample: %i"%np.array(y_test).sum())
        ax.bar_label(r2,padding=+3)

        ax.set_xticks(ticks=[0,1],labels=labels)
        plt.legend()

        plt.savefig(os.path.join(outdir,"events_%s.pdf"%name))
        plt.savefig(os.path.join(outdir,"events_%s.png"%name))
        
def plot_energies(df,outdir,name,bins=50,rng=(0,800)):
    fig = plt.figure()
    df["sc_rawEnergy"].hist(bins=bins,range=rng,histtype="step",label="sc_rawEnergy")
    df["regressedEnergy"].hist(bins=bins,range=rng,histtype="step",label="regressedEnergy")
    df["eg_gen_energy"].hist(bins=bins,range=rng,histtype="step",label="genEnergy")
    df["old_regressedEnergy"].hist(bins=bins,range=rng,histtype="step",label="old_regressedEnergy")
    
    plt.legend()
    plt.savefig(os.path.join(outdir,"energies_%s.pdf"%name))
    plt.savefig(os.path.join(outdir,"energies_%s.png"%name))
    
def scatter(df,outdir,name,ls=["raw","old"]):
    gen = df["eg_gen_energy"]
    raw = df["sc_rawEnergy"]
    reg = df["regressedEnergy"]
    old = df["old_regressedEnergy"]
    fig = plt.figure()
    for l in ls:
        if l == "old":
            plt.scatter(gen,old,label="old")
        if l == "raw":
            plt.scatter(gen,raw,label="raw")
        if l == "reg":
            plt.scatter(gen,reg,label="reg")
    
    plt.xlabel(r"$E_{Gen} [GeV]$")
    plt.ylabel(r"$E_{Reco} [GeV]$")
    plt.legend()
    plt.savefig(os.path.join(outdir,"reco_vs_gen_%s.pdf"%name))
    plt.savefig(os.path.join(outdir,"reco_vs_gen_%s.png"%name))
    
    
def plot_fractions(df,outdir,name,ls=["raw","old"]):
    frac_raw = df["frac_rawEnergy_genEnergy"]
    frac_reg = df["frac_regEnergy_genEnergy"]
    frac_old_reg = df["frac_old_regEnergy_genEnergy"]
    gen = df["eg_gen_energy"]

    fig = plt.figure()
    for l in ls:
        if l == "raw":
            plt.scatter(gen,frac_raw,label="sc_rawEnergy")
        if l == "old":
            plt.scatter(gen,frac_old_reg,label="old_regEnergy")
        if l == "reg":
            plt.scatter(gen,frac_old_reg,label="regEnergy")
    #plt.ylim([0,3])
    plt.xlim([0,1000])
    plt.legend()
    plt.savefig(os.path.join(outdir,"frac_vs_gen_%s.pdf"%name))
    plt.savefig(os.path.join(outdir,"frac_vs_gen_%s.png"%name))
    
    
def find_range(dfs,key):
    minimum = 9999999
    maximum = -9999999
    
    for df in dfs:
        if minimum > df[key].min():
            minimum = df[key].min()
        if maximum < df[key].max():
            maximum = df[key].max()
    return (minimum,maximum)

def plot_feature_hists(dfs,labels,outdir,density=True):
    bins = 50
    keys = dfs[0].keys()

    for key in keys:
        rng = find_range(dfs,key)
        if "frac" in key:
            rng = [0.75,1.5]
        fig = plt.figure()
        for df,label in zip(dfs,labels):
            df[key].hist(bins=bins,range=rng,density=density,label=label,histtype="step")
        plt.legend()
        plt.xlabel(key)
        plt.ylabel("a.u.")
        plt.title(key)
        plt.savefig(os.path.join(outdir,"hist_%s.pdf"%key))
        plt.savefig(os.path.join(outdir,"hist_%s.png"%key))
        plt.close()
        
COLORS = ["red","blue"]

def plot_feature_correlation(dfs,labels,outdir,target="eg_gen_energy",keys=None,save=True):

    if keys == None:
        keys = dfs[0].keys()
        
    for key in keys:
        fig = plt.figure()
        rng = find_range(dfs,key)

        fig, ax = plt.subplots()
        patches = []
        for i,df in enumerate(dfs):
            sns.histplot(data=df,x=target,y=key,ax=ax,color=COLORS[i],alpha=0.5,bins=(50,50),binrange=((0,1000),(rng)),label=labels[i])
        # Manually create legend
            patches.append(mpatches.Patch(color=COLORS[i], alpha=0.5, label=labels[i]))
        plt.legend(handles=patches)
        
        if save == True:
            plt.savefig(os.path.join(outdir,"corr_%s.pdf"%key))
            plt.savefig(os.path.join(outdir,"corr_%s.png"%key))    