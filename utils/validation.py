import hist
import awkward as ak
import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep
plt.style.use(hep.style.CMS)
from pathlib import Path
import pickle
import os
import torch

from .fit import fitCruijff, cruijff, get_mean_and_std

import pdb


def createHists():
    max_E, max_E_tot = 100, 600
    min_ratio = 0.5
    max_ratio = 2
    bins = 200

    return dict(
        h_pred = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="pred_energy", label="Predicted trackster energy (GeV)")),
        h_reco = hist.Hist(hist.axis.Regular(bins, 0., max_E, name="reco_energy", label="Trackster raw energy (GeV)")),
        h_reco_tot = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="reco_energy_tot", label="Total trackster raw energy (GeV)")),
        h_pred_tot = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="pred_energy_tot", label="Predicted energy for full endcap (GeV)")),
        h_cp = hist.Hist(hist.axis.Regular(bins, 0., max_E_tot, name="gen_energy", label="CaloParticle (true) energy (GeV)")),
        h_reco_tot_over_cp = hist.Hist(hist.axis.Regular(bins, min_ratio, max_ratio, name="reco_tot_over_cp", label="Total trackster raw energy / CaloParticle energy")),
        h_pred_tot_over_cp = hist.Hist(hist.axis.Regular(bins, min_ratio, max_ratio, name="pred_tot_over_cp", label="Total trackster predicted energy / CaloParticle energy")),
        h_cp_over_pred = hist.Hist(hist.axis.Regular(bins, min_ratio, max_ratio, name="cp_over_pred", label="CaloParticle energy / Total trackster raw energy")),
        h_cp_over_reco = hist.Hist(hist.axis.Regular(bins, min_ratio, max_ratio, name="cp_over_reco", label="CaloParticle energy / Total trackster predicted energy "))
    )

def createHists2D():
    hists = [] 
    bins = 200
    max_E = 610

    return dict(h_2d = hist.Hist(hist.axis.Regular(200,0,max_E, name="pred_energy", label="Predicted SC energy (GeV)"),
                                 hist.axis.Regular(200,0,max_E, name="true_energy", label="True SC energy (GeV)")
                                ),
               h_2d_target_v_pred = hist.Hist(hist.axis.Regular(50,0.5,2, name="target", label="Target"),
                                              hist.axis.Regular(50,0.5,2, name="prediction", label="Prediction")
                                             ),
                h_2d_target_v_nrHitsThreshold = hist.Hist(hist.axis.Regular(bins,0,4,name="target",label="Target"),
                                                          hist.axis.Regular(bins,0,4,name="nrHitsThreshold",label="nrHitsThreshold")
                                                         ),
                h_2d_target_v_eta = hist.Hist(hist.axis.Regular(bins,-3,3,name="target",label="Target"),
                                                hist.axis.Regular(bins,-3,3,name="eta",label="Eta")
                                                            ),
                h_2d_target_v_rawEnergy = hist.Hist(hist.axis.Regular(bins,0,100,name="target",label="Target"),
                                                hist.axis.Regular(bins,0,100,name="sc_rawEnergy",label="Raw Energy (GeV)")
                                                            ),
                h_2d_target_v_phiWidth = hist.Hist(hist.axis.Regular(bins,0,4,name="target",label="Target"),
                                                hist.axis.Regular(bins,0,4,name="phiWidth",label="Phi Width")
                                                            ),
                h_2d_target_v_rvar = hist.Hist(hist.axis.Regular(bins,0,4,name="target",label="Target"),
                                                hist.axis.Regular(bins,0,4,name="rvar",label="Rvar")
                                                            ),
                h_2d_target_v_numberOfSubClusters = hist.Hist(hist.axis.Regular(bins,0,4,name="target",label="Target"),
                                                hist.axis.Regular(bins,0,4,name="numberOfSubClusters",label="Number of SubClusters")
                                                            ),
                h_2d_target_v_clusterMaxDR = hist.Hist(hist.axis.Regular(bins,0,4,name="target",label="Target"),
                                                hist.axis.Regular(bins,0,4,name="clusterMaxDR",label="Cluster Max DR")  
               )
            )

# FEATURES = {"nrHitsThreshold":0,
#             "eta":1,
#             "sc_rawEnergy":2,
#             "phiWidth":3,
#             "rvar":4,
#             "numberOfSubClusters":5,
#             "clusterMaxDR":6
#            }

def writeHistograms(raw_energy,reg_energy,gen_energy,hists): # prediction = E_pred/E_reco, y = E_true/E_reco
    hists["h_reco_tot"].fill(raw_energy)
    hists["h_pred_tot"].fill(reg_energy)
    hists["h_cp"].fill(gen_energy)
    hists["h_cp_over_reco"].fill(gen_energy/raw_energy)
    hists["h_cp_over_pred"].fill(gen_energy/reg_energy)
    hists["h_pred_tot_over_cp"].fill(reg_energy/gen_energy)
    hists["h_reco_tot_over_cp"].fill(raw_energy/gen_energy)
    return hists

def writeHistograms2D(reg_energy,gen_energy,hists): # prediction = E_pred/E_reco, y = E_true/E_reco
    hists["h_2d"].fill(reg_energy,gen_energy)
    # hists["h_2d_target_v_pred"].fill(y,prediction)
    # hists["h_2d_target_v_nrHitsThreshold"].fill(y,X[feature_map["nrHitsThreshold"]])
    # hists["h_2d_target_v_eta"].fill(y,X[feature_map["eta"]])
    # hists["h_2d_target_v_rawEnergy"].fill(y,X[feature_map["sc_rawEnergy"]])
    # hists["h_2d_target_v_phiWidth"].fill(y,X[feature_map["phiWidth"]])
    # hists["h_2d_target_v_rvar"].fill(y,X[feature_map["rvar"]])
    # hists["h_2d_target_v_numberOfSubClusters"].fill(y,X[feature_map["numberOfSubClusters"]])
    # hists["h_2d_target_v_clusterMaxDR"].fill(y,X[feature_map["clusterMaxDR"]])
    return hists
    

def plotTracksterEnergies(hists):
    fig = plt.figure(figsize=(9, 9))
    hep.histplot([hists["h_reco"], hists["h_pred"]], label=["Raw trackster energy", "Predicted trackster energy"])
    plt.ylabel("Tracksters")
    plt.xlabel("Trackster energy (GeV)")
    plt.xlim(0, 50)
    plt.legend(loc="upper right")
    return fig

def plotFullEnergies(hists):
    fig = plt.figure()
    hep.histplot([hists["h_reco_tot"], hists["h_pred_tot"], hists["h_cp"]], yerr=False, label=["Raw SC energies", "Predicted SC energies", "CaloParticle (true) energy"])
    plt.ylabel("Events")
    plt.xlabel("Energy in endcap (GeV)")
    plt.legend()
    return [fig]


def plotTargetPred2D(hists):
    fig = plt.figure()
    hep.hist2dplot(hists["h_2d_target_v_pred"])
    plt.ylabel("Prediction")
    plt.xlabel("Target")
    return [fig]

def plotTargetFeatures2D(hists):
    figs = []
    for key in hists.keys():
        fig = plt.figure()
        hep.hist2dplot(hists[key])
        plt.ylabel(hists[key].axes[1].name)
        plt.xlabel("Target")
        figs.append(fig)
    return figs
        # plt.savefig(f"target_v_{feature}.png")
        # plt.close(fig)

def plotTargetPred(hists,fit=False):
    fig = plt.figure()
    hep.histplot([hists["h_cp_over_reco"], hists["h_cp_over_pred"]], yerr=False, label=["Target", "Prediction"])
    
    def plotFit(h:hist.Hist):
        fitRes = fitCruijff(h)
        params = fitRes.params
        x_plotFct = np.linspace(h.axes[0].centers[0], h.axes[0].centers[-1],500)
        plt.plot(x_plotFct,cruijff(x_plotFct,*params.makeTuple()), 
            label=f"Cruijff fit\n$\sigma={(params.sigmaL+params.sigmaR)/2:.3f}$, $\mu={params.m:.3f}$, " +r"$\frac{\sigma}{\mu}=" + f"{(params.sigmaL+params.sigmaR)/(2*params.m):.3f}$")

    if fit:
        plotFit(hists["h_cp_over_reco"])
        plotFit(hists["h_cp_over_pred"])

    plt.ylabel("Events")
    plt.xlabel("Ratio CP over Reco Energy")
    plt.legend()
    return [fig]

def plotRatioOverCP(hists,feature_map=None,fit=False):
    fig = plt.figure()
    hep.histplot([hists["h_reco_tot_over_cp"], hists["h_pred_tot_over_cp"]], yerr=False, label=["Raw SC energy fraction", "Predicted SC energy fraction"])
    
    def plotFit(h:hist.Hist):
        fitRes = fitCruijff(h)
        params = fitRes.params
        x_plotFct = np.linspace(h.axes[0].centers[0], h.axes[0].centers[-1],500)
        plt.plot(x_plotFct,cruijff(x_plotFct,*params.makeTuple()), 
            label=f"Cruijff fit\n$\sigma={(params.sigmaL+params.sigmaR)/2:.3f}$, $\mu={params.m:.3f}$, " +r"$\frac{\sigma}{\mu}=" + f"{(params.sigmaL+params.sigmaR)/(2*params.m):.3f}$")

    if fit:
        plotFit(hists["h_reco_tot_over_cp"])
        plotFit(hists["h_pred_tot_over_cp"])

    plt.ylabel("Events")
    plt.xlabel("Ratio over CaloParticle energy")
    plt.legend()
    return [fig]


def plotTrueVPred(hist2d):
    fig = plt.figure()
    hep.hist2dplot(hist2d["h_2d"],cmin=1)
    return [fig]

def createHistsParameter():
    boundary_min = 0
    boundary_max = 4
    bins = 200

    return dict(
        h_mu = hist.Hist(hist.axis.Regular(bins, 0.5, 2, name="mu", label="mu")),
        h_width = hist.Hist(hist.axis.Regular(bins, 0,0.5 , name="width", label="width")),
        h_a1 = hist.Hist(hist.axis.Regular(bins, boundary_min,boundary_max, name="a1", label="a1")),
        h_a2 = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="a2", label="a2")),
        h_p1 = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="p1", label="p1")),
        h_p2 = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="p2", label="p2")),
    )

def createHistsParameter2D(feature_map,normalization):
    boundary_min = 0
    boundary_max = 4
    bins = 50
    d = {}
    y_axis = hist.axis.Regular(bins,0,2,name="tgt",label="tgt")

    d["h_mu_tgt"] = hist.Hist(hist.axis.Regular(bins, 0.5, 2, name="mu", label="mu"),y_axis)
    d["h_width_tgt"] = hist.Hist(hist.axis.Regular(bins, 0,5 , name="width", label="width"),y_axis)
    d["h_a1_tgt"] = hist.Hist(hist.axis.Regular(bins, boundary_min,boundary_max, name="a1", label="a1"),y_axis)
    d["h_a2_tgt"] = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="a2", label="a2"),y_axis)
    d["h_p1_tgt"] = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="p1", label="p1"),y_axis)
    d["h_p2_tgt"] = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="p2", label="p2"),y_axis)

    y_axis = hist.axis.Regular(bins,0,2,name="pred",label="pred") # I'm using the normalized inputs but multiplied with the correction factor

    d["h_mu_pred"] = hist.Hist(hist.axis.Regular(bins, 0.5, 2, name="mu", label="mu"),y_axis)
    d["h_width_pred"] = hist.Hist(hist.axis.Regular(bins, 0,0.5 , name="width", label="width"),y_axis)
    d["h_a1_pred"] = hist.Hist(hist.axis.Regular(bins, boundary_min,boundary_max, name="a1", label="a1"),y_axis)
    d["h_a2_pred"] = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="a2", label="a2"),y_axis)
    d["h_p1_pred"] = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="p1", label="p1"),y_axis)
    d["h_p2_pred"] = hist.Hist(hist.axis.Regular(bins, boundary_min, boundary_max, name="p2", label="p2"),y_axis)



    if normalization == "MinMax":
        feature_values_min = 0
        feature_values_max = 1

    for key in feature_map.keys():
        x_axis = hist.axis.Regular(bins,feature_values_min,feature_values_max,name=key,label=key)

        d["h_mu_%s"%key] = hist.Hist(x_axis,hist.axis.Regular(bins, 0.5, 2, name="mu", label="mu"))
        d["h_width_%s"%key] = hist.Hist(x_axis,hist.axis.Regular(bins, 0,0.5 , name="width", label="width"))
        d["h_a1_%s"%key] = hist.Hist(x_axis,hist.axis.Regular(bins, boundary_min,boundary_max, name="a1", label="a1"))
        d["h_a2_%s"%key] = hist.Hist(x_axis,hist.axis.Regular(bins, boundary_min, boundary_max, name="a2", label="a2"))
        d["h_p1_%s"%key] = hist.Hist(x_axis,hist.axis.Regular(bins, boundary_min, boundary_max, name="p1", label="p1"))
        d["h_p2_%s"%key] = hist.Hist(x_axis,hist.axis.Regular(bins, boundary_min, boundary_max, name="p2", label="p2"))

    return d

def writeHistogramsParameter2D(parameters,hists,tgt,inpt,output,feature_map): #inpt should be normalized
    
    hists["h_mu_tgt"].fill(parameters[0],tgt)
    hists["h_width_tgt"].fill(parameters[1],tgt)
    hists["h_a1_tgt"].fill(parameters[2],tgt)
    hists["h_a2_tgt"].fill(parameters[3],tgt)
    hists["h_p1_tgt"].fill(parameters[4],tgt)
    hists["h_p2_tgt"].fill(parameters[5],tgt)

    hists["h_mu_pred"].fill(parameters[0],output*inpt[:,feature_map["sc_rawEnergy"]])
    hists["h_width_pred"].fill(parameters[1],output*inpt[:,feature_map["sc_rawEnergy"]])
    hists["h_a1_pred"].fill(parameters[2],output*inpt[:,feature_map["sc_rawEnergy"]])
    hists["h_a2_pred"].fill(parameters[3],output*inpt[:,feature_map["sc_rawEnergy"]])
    hists["h_p1_pred"].fill(parameters[4],output*inpt[:,feature_map["sc_rawEnergy"]])
    hists["h_p2_pred"].fill(parameters[5],output*inpt[:,feature_map["sc_rawEnergy"]])

    
    for key in feature_map.keys():
        hists["h_mu_%s"%key].fill(inpt[:,feature_map[key]],parameters[0])
        hists["h_width_%s"%key].fill(inpt[:,feature_map[key]],parameters[1])
        hists["h_a1_%s"%key].fill(inpt[:,feature_map[key]],parameters[2])
        hists["h_a2_%s"%key].fill(inpt[:,feature_map[key]],parameters[3])
        hists["h_p1_%s"%key].fill(inpt[:,feature_map[key]],parameters[4])
        hists["h_p2_%s"%key].fill(inpt[:,feature_map[key]],parameters[5])
    return hists

def writeHistogramsParameter(parameters,hists):
    hists["h_mu"].fill(parameters[0])
    hists["h_width"].fill(parameters[1])
    hists["h_a1"].fill(parameters[2])
    hists["h_a2"].fill(parameters[3])
    hists["h_p1"].fill(parameters[4])
    hists["h_p2"].fill(parameters[5])
    return hists

def plotParameter(hist,xlabel,ylabel="Events"):
    fig = plt.figure()
    hep.histplot(hist, yerr=False, label=xlabel)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    return [fig]

def plotParameter2D(hist):
    xlabel = hist.axes[0].name
    ylabel = hist.axes[1].name
    fig = plt.figure()
    hep.hist2dplot(hist,cmin=1, label=xlabel)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    return fig

plotsToSave = [plotFullEnergies, plotRatioOverCP]

def doFullValidation(X, y, prediction,model_path):
    hists = write_histograms(X, y, prediction)
    with open(os.path.join(model_path,"hists.pkl"), "wb") as f:
        pickle.dump(hists, f)
    
    for plotFct in plotsToSave:
        plotFct(hists)
        plt.savefig(os.path.join(model_path,plotFct.__name__ + ".png"))



def create_histograms(energies,feature_map,normalization):
    # Create Histograms
    hists = createHists()
    hists2D = createHists2D()
    histsParam = createHistsParameter()
    histsParam2D = createHistsParameter2D(feature_map,normalization)

    energy_labels = ["%s_%s"%(energies[i],energies[i+1]) for i in range(len(energies)-1)]
    
    hists_energies = []
    histsParam_energies = []
    histsParam2D_energies = []
    for i in range(len(energies)-1):
        hists_energies.append(createHists())
        histsParam_energies.append(createHistsParameter())
        histsParam2D_energies.append(createHistsParameter2D(feature_map,normalization))

    d = {"hists":hists,
        "hists2D":hists2D,
        "histsParam":histsParam,
        "histsParam2D":histsParam2D,
        "hists_energies":hists_energies,
        "histsParam_energies":histsParam_energies,
        "histsParam2D_energies":histsParam2D_energies}
    return d
        
        
def fill_histograms(d_hists,inpt,tgt,raw_energy,pred_energy,gen_energy,dscb,energies,features):
    
    
    output = torch.exp(-dscb.mu).detach().cpu()
    tgt = torch.exp(-tgt.cpu())
    #gen_energy = tgt.T*inpt_orig[:,features["sc_rawEnergy"]].flatten()
    
    # Fill hists
    d_hists["hists"] = writeHistograms(raw_energy,pred_energy,gen_energy,d_hists["hists"])
    d_hists["hists2D"] = writeHistograms2D(pred_energy,gen_energy,d_hists["hists2D"])

    params = [torch.exp(dscb.mu.detach().cpu().squeeze()),
              dscb.sigma.detach().cpu().squeeze(),
              dscb.alphaL.detach().cpu().squeeze(),
              dscb.alphaR.detach().cpu().squeeze(),
              dscb.etaL.detach().cpu().squeeze(),
              dscb.etaR.detach().cpu().squeeze()]
    
    d_hists["histsParam"] = writeHistogramsParameter(params,d_hists["histsParam"])
    d_hists["histsParam2D"] = writeHistogramsParameter2D(params,d_hists["histsParam2D"],torch.exp(tgt).flatten(),inpt,output.flatten(),features)

    for i in range(len(energies)-1):
        mask = torch.where((gen_energy >= energies[i]) & (gen_energy<energies[i+1]))[0]
        d_hists["hists_energies"][i] = writeHistograms(raw_energy[mask],pred_energy[mask],gen_energy[mask],d_hists["hists_energies"][i])

        params = [torch.exp(dscb.mu[mask].detach().cpu().squeeze()),
                  dscb.sigma[mask].detach().cpu().squeeze(),
                  dscb.alphaL[mask].detach().cpu().squeeze(),
                  dscb.alphaR[mask].detach().cpu().squeeze(),
                  dscb.etaL[mask].detach().cpu().squeeze(),
                  dscb.etaR[mask].detach().cpu().squeeze()]
        
        d_hists["histsParam_energies"][i] = writeHistogramsParameter(params,d_hists["histsParam_energies"][i])
        d_hists["histsParam2D_energies"][i] = writeHistogramsParameter2D(params,d_hists["histsParam2D_energies"][i],torch.exp(tgt[mask]).flatten(),inpt[mask],output[mask].flatten(),features)

    return d_hists  

def plot_histograms(hists,plotsToSave,name="",epoch=None,comet=None,outDir=None):
    for plotFct in plotsToSave:
        figs = plotFct(hists)
        for fig in figs:
            if comet:
                comet.log_figure(plotFct.__name__+name,fig,step=epoch)
                plt.close(fig)
            else:
                fig.savefig(os.path.join(outDir,plotFct.__name__ + "%s.png"%name))
                plt.close(fig)

def plot_parameters(hists,name="Params",epoch=None,comet=None,outDir=None):
    for key in hists.keys():
        fig = plotParameter(hists[key],key)[0]
        if comet:   
            comet.log_figure("%s_%s.png"%(name,key),fig,step=epoch)
            plt.close(fig)
        else:
            fig.savefig(os.path.join(outDir,"%s_%s.png"%(name,key)))
            plt.close(fig)

def plot_parameters2D(hists,name="Params",epoch=None,comet=None,outDir=None):
    for key in hists.keys():
        fig = plotParameter2D(hists[key])
        if comet:   
            comet.log_figure("%s_%s.png"%(name,key),fig,step=epoch)
            plt.close(fig)
        else:
            k = key.replace("/","_")
            fig.savefig(os.path.join(outDir,"%s_%s.png"%(name,k)))
            plt.close(fig)


def get_cruijff_mean_sigma_from_hists(hists,histtypes=["h_reco_tot_over_cp","h_pred_tot_over_cp"]):

    d = {}
    for histtype in histtypes:
        t = histtype.split("_")[1]
        d["mu_%s"%t] = []
        d["mu_err_%s"%t] = []
        d["sig_%s"%t] = []
        d["sig_err_%s"%t] = []
        d["res_%s"%t] = []
        d["res_err_%s"%t] = []

    for h in hists:
        for histtype in histtypes: 
            t = histtype.split("_")[1]
            fitRes = fitCruijff(h[histtype])
            params = fitRes.params
            cov = fitRes.covMatrix

            mu = params.m
            mu_err = np.sqrt(cov[0,0])
            sigma = 0.5*(params.sigmaL+params.sigmaR)            
            sigma_err = np.sqrt(cov[1,1]+cov[2,2]+2*cov[2,3]/4)
            covL_ = cov[1,2]
            covR_ = cov[2,3]
            cov = (covR_+covL_)/2
            
            res = sigma/mu
            res_err = sigma/mu*np.sqrt((mu_err/mu)**2+(sigma_err/sigma)**2-2*(cov)/(sigma*mu))

            d["mu_%s"%t].append(mu)
            #d["mu_err_%s"%t].append(mu_err)
            d["mu_err_%s"%t].append(0)
            d["sig_%s"%t].append(sigma)
            #d["sig_err_%s"%t].append(sigma_err)
            d["sig_err_%s"%t].append(0)
            d["res_%s"%t].append(res)
            #d["res_err_%s"%t].append(res_err)
            d["res_err_%s"%t].append(0)


    for key in d.keys():
        d[key] = np.array(d[key])
    return d


def create_parameter_plots(d,x_axis,x_axis_uncert,xaxis_label,root):
    fig = plt.figure()
    plt.errorbar(x_axis,d["mu_reco"],xerr=x_axis_uncert,yerr=d["mu_err_reco"],linestyle="",marker="x",color="red",label="Reconstructed")
    plt.errorbar(x_axis,d["mu_pred"],xerr=x_axis_uncert,yerr=d["mu_err_pred"],linestyle="",marker=".",color="blue",label="Predicted")
    plt.axhline(1,color="black",linestyle="--")
    plt.legend()
    fig.savefig(os.path.join(root,"m_v_%s.png"%xaxis_label))

    fig = plt.figure()
    plt.errorbar(x_axis,d["sig_reco"],xerr=x_axis_uncert,yerr=d["sig_err_pred"],linestyle="",marker="x",color="red",label="Reconstructed")
    plt.errorbar(x_axis,d["sig_pred"],xerr=x_axis_uncert,yerr=d["sig_err_pred"],linestyle="",marker=".",color="blue",label="Predicted")
    plt.legend()
    fig.savefig(os.path.join(root,"sig_v_%s.png"%xaxis_label))

    fig = plt.figure()
    plt.errorbar(x_axis,d["res_reco"],xerr=x_axis_uncert,yerr=d["res_err_pred"],linestyle="",marker="x",color="red",label="Reconstructed")
    plt.errorbar(x_axis,d["res_pred"],xerr=x_axis_uncert,yerr=d["res_err_pred"],linestyle="",marker=".",color="blue",label="Predicted")
    plt.legend()
    fig.savefig(os.path.join(root,"res_v_%s.png"%xaxis_label))
    plt.close("all")

def produce_plots(d_hists,energies,outDir):
    # 1D hists
    plotsToSave = [plotFullEnergies, plotRatioOverCP]
    plot_histograms(d_hists["hists"],plotsToSave,plotsToSave,outDir=outDir)
    for i in range(len(energies)-1):
        plot_histograms(d_hists["hists_energies"][i],plotsToSave,energies[i],outDir=outDir)

    # 2D hists
    plotsToSave = [plotTrueVPred]
    plot_histograms(d_hists["hists2D"],plotsToSave,outDir=outDir)

    # Parameters
    plot_parameters(d_hists["histsParam"],outDir=outDir)
    for i in range(len(energies)-1):
        plot_parameters(d_hists["histsParam_energies"][i],energies[i],outDir=outDir)

    plot_parameters2D(d_hists["histsParam2D"],outDir=outDir)
    # for i in range(len(energies)-1):
    #     plot_parameters2D(d_hists["histsParam2D_energies"][i],energies[i],outDir=outDir)

    # Parameters from Cruijff fit
    cruijff_params_uncertainty = get_cruijff_mean_sigma_from_hists(d_hists["hists_energies"])
    energies = np.array(energies)
    x_axis = (energies[1:]+energies[:-1])/2
    x_axis_uncert = energies[1:]-x_axis
    create_parameter_plots(cruijff_params_uncertainty,x_axis,x_axis_uncert,"energy",outDir)

    # Compare per-object resolution with Cruijff fit
    params_uncertainty = get_parameters_from_hists(d_hists["histsParam_energies"])
    create_comparison_parameter_plots(params_uncertainty,cruijff_params_uncertainty,x_axis,x_axis_uncert,"energy",outDir)
        

def get_parameters_from_hists(hists):
    d = {}
    d["per_object_width"] = []
    d["per_object_width_err"] = []
    for h in hists:
        mean,std = get_mean_and_std(h["h_width"])
        d["per_object_width"].append(mean)
        d["per_object_width_err"].append(std)
    return d

def create_comparison_parameter_plots(d_pred,d_cruijff,x_axis,x_axis_uncert,xaxis_label,root):
    fig = plt.figure()
    plt.errorbar(x_axis,d_pred["per_object_width"],xerr=x_axis_uncert,yerr=d_pred["per_object_width_err"],linestyle="",marker="x",color="red",label="Per object width")
    plt.errorbar(x_axis,d_cruijff["sig_pred"],xerr=x_axis_uncert,yerr=d_cruijff["sig_err_pred"],linestyle="",marker=".",color="blue",label="Cruijff sigma")
    plt.legend()
    fig.savefig(os.path.join(root,"comparison_width_v_%s.png"%xaxis_label))
    plt.close("all")