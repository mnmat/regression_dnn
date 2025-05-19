import os
import awkward as ak
import numpy as np
import pickle


def count_parameters(model):
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return params

def create_dirs(path,add_php=False):
    if not os.path.exists(path):
        os.makedirs(path)
    if add_php:
        cmd = "cp /eos/home-m/mmatthew/www/index/index.php %s"%path
        os.system(cmd)

def get_training_test_files(path,training_dir="s4Flat_genMatched",test_dir="s5Reg_genMatched"):
    training_path = os.path.join(path,training_dir)
    test_path = os.path.join(path,test_dir)

    training_file = "HLTAnalyzerTree_IDEAL_Flat_train.root"
    test_file = "Run3HLT_IdealIC_IdealTraining_stdVar_stdCuts_ntrees1500_applied.root"

    return os.path.join(training_path,training_file), os.path.join(test_path,test_file)

def modify_tree(df):
    df["old_regressedEnergy"] = df["regressedEnergy"]
    df["regressedEnergy"] = df["rawEnergy"]*df["regEEMean"]
    
    df["frac_rawEnergy_genEnergy"] = df["rawEnergy"]/df["eg_gen_energy"]
    df["frac_old_regEnergy_genEnergy"] = df["old_regressedEnergy"]/df["eg_gen_energy"]
    df["frac_regEnergy_genEnergy"] = df["regressedEnergy"]/df["eg_gen_energy"]
    
    df["old_regEEMean"] = df["old_regressedEnergy"]/df["rawEnergy"]
    return df


# Get Variables

def get_mask(sc):
    mask = sc.isEB
    mask = ak.values_astype(mask, np.int32)
    isEB = ak.where(mask==1)[0]
    isEE = ak.where(mask==0)[0]
    return isEB, isEE

def get_variables(tree):
   
    sc = tree["sc"].array()
    ssFrac = tree["ssFrac"].array()
    clus1 = tree["clus1"].array()
    clus2 = tree["clus2"].array()
    clus3 = tree["clus3"].array()
    nrVert = tree["nrVert"].array()

    # isEB

    isEB, isEE = get_mask(sc)

    # nrVert Variables

    nrVert = nrVert

    # sc Variables
    sc_rawEnergy = sc["rawEnergy"]
    sc_etaWidth = sc["etaWidth"]
    sc_phiWidth = sc["phiWidth"]
    sc_seedClusEnergy = sc["seedClusEnergy"]
    sc_numberOfSubClusters = sc["numberOfSubClusters"]
    sc_clusterMaxDR = sc["clusterMaxDR"]
    sc_clusterMaxDRDPhi = sc["clusterMaxDRDPhi"]
    sc_clusterMaxDRDEta = sc["clusterMaxDRDEta"]
    sc_clusterMaxDRRawEnergy = sc["clusterMaxDRRawEnergy"]
    sc_iEtaOrX = sc["iEtaOrX"]
    sc_iPhiOrY = sc["iPhiOrY"]
    sc_seedEta = sc["seedEta"]

    # ssFrac Variables

    ssFrac_e3x3 = ssFrac["e3x3"]
    ssFrac_e2nd = ssFrac["e2nd"]
    ssFrac_eLeftRightDiffSumRatio = ssFrac["eLeftRightDiffSumRatio"] 
    ssFrac_eTopBottomDiffSumRatio = ssFrac["eTopBottomDiffSumRatio"]
    ssFrac_sigmaIEtaIEta = ssFrac["sigmaIEtaIEta"]
    ssFrac_sigmaIEtaIPhi = ssFrac["sigmaIEtaIPhi"]
    ssFrac_sigmaIPhiIPhi = ssFrac["sigmaIPhiIPhi"]
    ssFrac_eMax = ssFrac["eMax"]

    # clus123 Variables

    clus1_clusterRawEnergy = clus1["clusterRawEnergy"]
    clus2_clusterRawEnergy = clus2["clusterRawEnergy"]
    clus3_clusterRawEnergy = clus3["clusterRawEnergy"]

    clus1_clusterDPhiToSeed = clus1["clusterDPhiToSeed"]
    clus2_clusterDPhiToSeed = clus2["clusterDPhiToSeed"]
    clus3_clusterDPhiToSeed = clus3["clusterDPhiToSeed"]

    clus1_clusterDEtaToSeed = clus1["clusterDEtaToSeed"]
    clus2_clusterDEtaToSeed = clus2["clusterDEtaToSeed"]
    clus3_clusterDEtaToSeed = clus3["clusterDEtaToSeed"]


    # EE
    ee_nrVert = nrVert[isEE].tolist()
    ee_sc_rawEnergy = sc_rawEnergy[isEE].tolist()
    ee_sc_etaWidth = sc_etaWidth[isEE].tolist()
    ee_sc_phiWidth = sc_phiWidth[isEE].tolist()
    ee_ssFrac_e3x3_o_sc_rawEnergy = ssFrac_e3x3[isEE]/sc_rawEnergy[isEE].tolist()
    ee_sc_seedClusEnergy_o_sc_rawEnergy = sc_seedClusEnergy[isEE]/sc_rawEnergy[isEE].tolist()
    ee_ssFrac_eMax_o_sc_rawEnergy = ssFrac_eMax[isEE]/sc_rawEnergy[isEE].tolist()
    ee_ssFrac_e2nd_o_sc_rawEnergy = ssFrac_e2nd[isEE]/sc_rawEnergy[isEE].tolist()
    ee_ssFrac_eLeftRightDiffSumRatio = ssFrac_eLeftRightDiffSumRatio[isEE].tolist()
    ee_ssFrac_eTopBottomDiffSumRatio = ssFrac_eTopBottomDiffSumRatio[isEE].tolist()
    ee_ssFrac_sigmaIEtaIEta = ssFrac_sigmaIEtaIEta[isEE].tolist()
    ee_ssFrac_sigmaIEtaIPhi = ssFrac_sigmaIEtaIPhi[isEE].tolist()
    ee_ssFrac_sigmaIPhiIPhi = ssFrac_sigmaIPhiIPhi[isEE].tolist()
    ee_sc_numberOfSubClusters = sc_numberOfSubClusters[isEE].tolist()
    ee_sc_clusterMaxDR = sc_clusterMaxDR[isEE].tolist()
    ee_sc_clusterMaxDRDPhi = sc_clusterMaxDRDPhi[isEE].tolist()
    ee_sc_clusterMaxDRDEta = sc_clusterMaxDRDEta[isEE].tolist()
    ee_sc_clusterMaxDRRawEnergy_o_sc_rawEnergy = sc_clusterMaxDRRawEnergy[isEE]/sc_rawEnergy[isEE].tolist()
    ee_clus1_clusterRawEnergy_o_sc_rawEnergy = clus1_clusterRawEnergy[isEE]/sc_rawEnergy[isEE].tolist()
    ee_clus2_clusterRawEnergy_o_sc_rawEnergy = clus2_clusterRawEnergy[isEE]/sc_rawEnergy[isEE].tolist()
    ee_clus3_clusterRawEnergy_o_sc_rawEnergy = clus3_clusterRawEnergy[isEE]/sc_rawEnergy[isEE].tolist()
    ee_clus1_clusterDPhiToSeed = clus1_clusterDPhiToSeed[isEE].tolist()
    ee_clus2_clusterDPhiToSeed = clus2_clusterDPhiToSeed[isEE].tolist()
    ee_clus3_clusterDPhiToSeed = clus3_clusterDPhiToSeed[isEE].tolist()
    ee_clus1_clusterDEtaToSeed = clus1_clusterDEtaToSeed[isEE].tolist()
    ee_clus2_clusterDEtaToSeed = clus2_clusterDEtaToSeed[isEE].tolist()
    ee_clus3_clusterDEtaToSeed = clus3_clusterDEtaToSeed[isEE].tolist()
    ee_sc_iEtaOrX = sc_iEtaOrX[isEE].tolist()
    ee_sc_iPhiOrY = sc_iPhiOrY[isEE].tolist()
    ee_sc_seedEta = sc_seedEta[isEE].tolist()

    features_ee =  np.array(
        [ee_nrVert,
        ee_sc_rawEnergy,
        ee_sc_etaWidth,
        ee_sc_phiWidth,
        ee_ssFrac_e3x3_o_sc_rawEnergy,
        ee_sc_seedClusEnergy_o_sc_rawEnergy,
        ee_ssFrac_eMax_o_sc_rawEnergy,
        ee_ssFrac_e2nd_o_sc_rawEnergy,
        ee_ssFrac_eLeftRightDiffSumRatio,
        ee_ssFrac_eTopBottomDiffSumRatio,
        ee_ssFrac_sigmaIEtaIEta,
        ee_ssFrac_sigmaIEtaIPhi,
        ee_ssFrac_sigmaIPhiIPhi,
        ee_sc_numberOfSubClusters,
        ee_sc_clusterMaxDR,
        ee_sc_clusterMaxDRDPhi,
        ee_sc_clusterMaxDRDEta,
        ee_sc_clusterMaxDRRawEnergy_o_sc_rawEnergy,
        ee_clus1_clusterRawEnergy_o_sc_rawEnergy,
        ee_clus2_clusterRawEnergy_o_sc_rawEnergy,
        ee_clus3_clusterRawEnergy_o_sc_rawEnergy,
        ee_clus1_clusterDPhiToSeed,
        ee_clus2_clusterDPhiToSeed,
        ee_clus3_clusterDPhiToSeed,
        ee_clus1_clusterDEtaToSeed,
        ee_clus2_clusterDEtaToSeed,
        ee_clus3_clusterDEtaToSeed,
        ee_sc_iEtaOrX,
        ee_sc_iPhiOrY,
        ee_sc_seedEta]
    ).T

    # EB
    eb_nrVert = nrVert[isEB].tolist()
    eb_sc_rawEnergy = sc_rawEnergy[isEB].tolist()
    eb_sc_etaWidth = sc_etaWidth[isEB].tolist()
    eb_sc_phiWidth = sc_phiWidth[isEB].tolist()
    eb_ssFrac_e3x3_o_sc_rawEnergy = ssFrac_e3x3[isEB]/sc_rawEnergy[isEB].tolist()
    eb_sc_seedClusEnergy_o_sc_rawEnergy = sc_seedClusEnergy[isEB]/sc_rawEnergy[isEB].tolist()
    eb_ssFrac_eMax_o_sc_rawEnergy = ssFrac_eMax[isEB]/sc_rawEnergy[isEB].tolist()
    eb_ssFrac_e2nd_o_sc_rawEnergy = ssFrac_e2nd[isEB]/sc_rawEnergy[isEB].tolist()
    eb_ssFrac_eLeftRightDiffSumRatio = ssFrac_eLeftRightDiffSumRatio[isEB].tolist()
    eb_ssFrac_eTopBottomDiffSumRatio = ssFrac_eTopBottomDiffSumRatio[isEB].tolist()
    eb_ssFrac_sigmaIEtaIEta = ssFrac_sigmaIEtaIEta[isEB].tolist()
    eb_ssFrac_sigmaIEtaIPhi = ssFrac_sigmaIEtaIPhi[isEB].tolist()
    eb_ssFrac_sigmaIPhiIPhi = ssFrac_sigmaIPhiIPhi[isEB].tolist()
    eb_sc_numberOfSubClusters = sc_numberOfSubClusters[isEB].tolist()
    eb_sc_clusterMaxDR = sc_clusterMaxDR[isEB].tolist()
    eb_sc_clusterMaxDRDPhi = sc_clusterMaxDRDPhi[isEB].tolist()
    eb_sc_clusterMaxDRDEta = sc_clusterMaxDRDEta[isEB].tolist()
    eb_sc_clusterMaxDRRawEnergy_o_sc_rawEnergy = sc_clusterMaxDRRawEnergy[isEB]/sc_rawEnergy[isEB].tolist()
    eb_clus1_clusterRawEnergy_o_sc_rawEnergy = clus1_clusterRawEnergy[isEB]/sc_rawEnergy[isEB].tolist()
    eb_clus2_clusterRawEnergy_o_sc_rawEnergy = clus2_clusterRawEnergy[isEB]/sc_rawEnergy[isEB].tolist()
    eb_clus3_clusterRawEnergy_o_sc_rawEnergy = clus3_clusterRawEnergy[isEB]/sc_rawEnergy[isEB].tolist()
    eb_clus1_clusterDPhiToSeed = clus1_clusterDPhiToSeed[isEB].tolist()
    eb_clus2_clusterDPhiToSeed = clus2_clusterDPhiToSeed[isEB].tolist()
    eb_clus3_clusterDPhiToSeed = clus3_clusterDPhiToSeed[isEB].tolist()
    eb_clus1_clusterDEtaToSeed = clus1_clusterDEtaToSeed[isEB].tolist()
    eb_clus2_clusterDEtaToSeed = clus2_clusterDEtaToSeed[isEB].tolist()
    eb_clus3_clusterDEtaToSeed = clus3_clusterDEtaToSeed[isEB].tolist()
    eb_sc_iEtaOrX = sc_iEtaOrX[isEB].tolist()
    eb_sc_iPhiOrY = sc_iPhiOrY[isEB].tolist()


    features_eb =  np.array(
        [eb_nrVert,
        eb_sc_rawEnergy,
        eb_sc_etaWidth,
        eb_sc_phiWidth,
        eb_ssFrac_e3x3_o_sc_rawEnergy,
        eb_sc_seedClusEnergy_o_sc_rawEnergy,
        eb_ssFrac_eMax_o_sc_rawEnergy,
        eb_ssFrac_e2nd_o_sc_rawEnergy,
        eb_ssFrac_eLeftRightDiffSumRatio,
        eb_ssFrac_eTopBottomDiffSumRatio,
        eb_ssFrac_sigmaIEtaIEta,
        eb_ssFrac_sigmaIEtaIPhi,
        eb_ssFrac_sigmaIPhiIPhi,
        eb_sc_numberOfSubClusters,
        eb_sc_clusterMaxDR,
        eb_sc_clusterMaxDRDPhi,
        eb_sc_clusterMaxDRDEta,
        eb_sc_clusterMaxDRRawEnergy_o_sc_rawEnergy,
        eb_clus1_clusterRawEnergy_o_sc_rawEnergy,
        eb_clus2_clusterRawEnergy_o_sc_rawEnergy,
        eb_clus3_clusterRawEnergy_o_sc_rawEnergy,
        eb_clus1_clusterDPhiToSeed,
        eb_clus2_clusterDPhiToSeed,
        eb_clus3_clusterDPhiToSeed,
        eb_clus1_clusterDEtaToSeed,
        eb_clus2_clusterDEtaToSeed,
        eb_clus3_clusterDEtaToSeed,
        eb_sc_iEtaOrX,
        eb_sc_iPhiOrY]
    ).T

    return features_ee, features_eb
    
def get_evt(tree):

    
    evt = f["egRegTree"]["evt"].array()
    runnr = evt["runnr"]
    lumiSec = evt["lumiSec"]
    eventnr = evt["eventnr"]
    
    sc = tree["sc"].array()
    isEB, isEE = get_mask(sc)


    ee_evt = np.array([
        runnr[isEE].tolist(),
        lumiSec[isEE].tolist(),
        eventnr[isEE].tolist(),
    ])
    eb_evt = np.array([
        runnr[isEB].tolist(),
        lumiSec[isEB].tolist(),
        eventnr[isEB].tolist(),
    ])
    
    return ee_evt, eb_evt

def get_gen_energy(tree):
    target_arr = tree["mc"].array(library="np")["energy"]
    sc = tree["sc"].array()
    isEB, isEE = get_mask(sc)
    
    return np.array(target_arr[isEE].tolist()), np.array(target_arr[isEB].tolist())
    