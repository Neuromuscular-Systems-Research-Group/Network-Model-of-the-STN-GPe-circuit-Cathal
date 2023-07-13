import numpy as np
import pickle
import os
import copy
from os.path import isfile, join
import networkx as nx
import random
from additional_functions import *
import time as time
from nxmetis import partition

import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from collections import Counter
from scipy import stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import copy
from os.path import isfile, join

import subprocess 

dataAlias = {i:val for i,val in enumerate(["sync","eigs","STNBeta","GPeBeta","STNMean","GPeMean","SAMPA","GAMPA", "SGABA","GGABA" ])}
nets= {0:"ImprovedSpatial",1:"ActionSelection",2:"Small_world",3:"Scale_free"};
ps = {"ImprovedSpatial":4.5,"ActionSelection":1.7,"Small_world":0.05,"Scale_free":1}


def get_spike_trains(pop):
      main_lis = []
      for sp in pop.spiketrains:
            s_lis = [ float(i) for i in sp] 
            main_lis.append(np.array(s_lis))

      return main_lis


def partitionGraph2(netname,noParts,amount,n):
    el = load_EL(netname)
    G = nx.Graph()
    G.add_edges_from([[e[0],e[1]] for e in el])
    edge_cut, parts = partition(G,noParts)

    withinGroup = [(i,i) for i in range(noParts)]
    betweenGroup = [(i,i+1) for i in range(noParts) if (i+1)<noParts]

    validEdges = []

    for a,b in withinGroup:
        for x in parts[a]:
            for y in parts[b]:
                if ((x,y) in el) or ((y,x) in el):
                    validEdges.append((x,y))
                    
    for a,b in betweenGroup:
        for x in parts[a]:
            for y in parts[b]:
                if ((x,y) in el) or ((y,x) in el):
                    if random.random()<amount:
                        validEdges.append((x,y))

    updatedEL = recalculatePynnWeights(validEdges,n)
    cutname = f"Cut_{netname}"
    np.save(cutname,updatedEL)
    return getNetDat(el,updatedEL,n)

def randomRemoval(netname,dud,removeSize,n):
    el = load_EL(netname)
    keep = int((1-removeSize)*len(el))
    indices = np.random.choice(range(len(el)),size=keep,replace=False)
    el2 = [(el[ind][0],el[ind][1])  for ind in indices]
    updatedEL = recalculatePynnWeights(el2,n)
    cutname = f"Rand_{netname}"
    np.save(cutname,updatedEL)
    return getNetDatRand(updatedEL,n)
    

def getNetDatRand(elPost,n):
    postSize = len(elPost)
    postlam2,postlamN = getEigs(elPost,2*n)
    dat = {"lam2":postlam2,"lamN":postlamN,"size":postSize}
    return dat

def savenet(net,netname,n,k,p,r,rtn=False):
    gen_functions = {   "Scale_free":SG_ScaleFree,
                        "Small_world": SG_SmallWorld,
                        "Spatial":SG_ExponentialSpatial,
                        "Regular":SG_Regular,
                        "SBlock":SG_SBlock,
                        "ImprovedSpatial":SG_SpatialImproved,
                        "Random":SG_ER}

    network_gen_function = gen_functions[net]
    STG_list,GTS_list,GTG_list,graph_measures,all_edges = network_gen_function(n,k,p,r,dev=True)
    np.save(netname,all_edges)
    if rtn:
        return all_edges
    
def plotRasterLFP(name):
    resultsLoc = os.path.join("Results",f"{name}_results.npy")
    results = np.load(resultsLoc,allow_pickle=True).item()
    cols = '#2A92F5'
    colg = '#F5352A'
    STNdat = results["STNdata"]
    GPedat = results["GPedata"]
    SLFP = np.array(STNdat.filter(name='soma(0.5).v')[0]).mean(axis=1)
    GLFP = np.array(GPedat.filter(name='soma(0.5).v')[0]).mean(axis=1)
    STrain =  STNdat.spiketrains
    GTrain = GPedat.spiketrains
    sync = float(np.sqrt(results["STN synchrony"]*results["GPe synchrony"]))
    textstr = r'Mean $\chi$: '+ f'{sync:>3.3f}' + f"\nSTN FR: {results['SMean']:>6.2f} \nGPe FR: {results['GMean']:>6.2f} " 

    props = dict(boxstyle='round', facecolor='white', alpha=1,edgecolor='white')
    # place a text box in upper left in axes coords
    infigfontsize = 12
    titlefontsize = 14
    ilim = max(results["networkInfo"]["nSTN"],results["networkInfo"]["nGPe"])
    simtime = results["simtime"]
    xlims = [0, simtime]
    x_axis = np.linspace(0,simtime,len(GLFP))

    fig,ax = plt.subplots(2,1,figsize=[9,5],sharex=True,dpi=120)
    for i, train in enumerate(GTrain):
            strainExists = False
            if i<len(STrain):
                strainExists = True
                train1 = STrain[i]

            train2 = train
            if i>ilim:
                break
            if i==0:
                ax[0].plot(train1, i*np.ones(len(train1)), 'o', color = cols,markersize=3, alpha=0.5, label='STN')
                ax[0].plot(train2, i*np.ones(len(train2)), 'o', color = colg,markersize=4, alpha=0.8, label='GPe')
            else:
                if strainExists:
                    ax[0].plot(train1, i*np.ones(len(train1)), 'o', color = cols,markersize=3, alpha=0.8) 
                ax[0].plot(train2, i*np.ones(len(train2)), 'o', color = colg,markersize=4, alpha=0.9)

    k = results["networkInfo"]["k"]
    #f'{netlab[data_dict["Network_type"]]} Network'
    ax[0].set_title(f'Simulation Raster Plot: k={k}',fontsize=titlefontsize)
    ax[0].set_ylabel("Neuron Index",fontsize=infigfontsize)

    ax[0].set_xlim(xlims)

    ax[0].set_ylim(-1,ilim+1)
    ax[0].legend(fontsize=infigfontsize,loc='upper right',framealpha=1)
    #ax[0].legend(fontsize=16,framealpha=1,shadow = True,borderpad=0.8,loc='upper right')

    ax[1].plot(x_axis, SLFP, label = 'STN',color = cols, alpha=0.9)
    ax[1].plot(x_axis, GLFP, label = 'GPe',color = colg, alpha=1)
    ax[1].set_ylabel("Membrane Potential (mV)",fontsize=infigfontsize)
    ax[1].set_xlabel("Time (ms)",fontsize=infigfontsize)
    ax[1].set(ylim=[-85,-30,])
    ax[1].text(0.72, 0.95, textstr, transform=ax[1].transAxes, color='k', fontsize=infigfontsize, verticalalignment='top', bbox=props)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.025, hspace=0.01)

def genCorticalEL(n):
    convergence = 20
    nCortical = convergence*n
    w = 0.25
    el = []
    for i in range(n):
        el.extend(list(np.array([np.random.choice(range(nCortical),size=convergence,replace=False),np.ones(convergence)*i,w*np.ones(convergence)]).T))

    el = [(int(e[0]),int(e[1]), e[2]) for e in el]
    return el

def initialiseNet(genFunc,n,k,p,plot = True, r= 1.0):
    gen_functions = {   "Scale_free":SG_ScaleFree,
                    "Small_world": SG_SmallWorld,
                    "Spatial":SG_ExponentialSpatial,
                    "Regular":SG_Regular,
                    "SBlock":SG_SBlock,
                    "ImprovedSpatial":SG_SpatialImproved,
                    "ActionSelection":SG_ActionSelection}
    
    thisGenFunc = gen_functions[genFunc]
    STG_list,GTS_list,GTG_list,graph_measures = thisGenFunc(n,k,p,r)
    CTS = genCorticalEL(n)
    graphData = {"SGel":STG_list,"GSel":GTS_list,"GGel":GTG_list,"CSel":CTS,
                 "graphMeasures":graph_measures,
                 "networkName":genFunc,
                 "nSTN":n,"nGPe":n, "k":k,"p":p,"r":r,"converg":20,
                 "nCTX":20*n}
    
    lsg = copy.deepcopy(STG_list)
    lgs = copy.deepcopy(GTS_list)
    lgg = copy.deepcopy(GTG_list)
    fullEL = update_index(lsg,0,n,0) + update_index(lgs,n,0,0) + update_index(lgg,n,n,0)

    return fullEL,graphData




def partitionGraph(fullEL,noParts,betweenGroupFactor = 1):
    G = nx.Graph()
    G.add_edges_from([[e[0],e[1]] for e in fullEL])
    edge_cut, parts = partition(G,noParts)
    withinGroup = [(i,i) for i in range(noParts)]
    betweenGroup = [(i,i+1) for i in range(noParts) if (i+1)<noParts]
    el = [(x[0],x[1]) for x in fullEL]
    validEdges = []

    for a,b in withinGroup:
        for x in parts[a]:
            for y in parts[b]:
                if ((x,y) in el):
                    validEdges.append((x,y))
                    
    for a,b in betweenGroup:
        for x in parts[a]:
            for y in parts[b]:
                if ((x,y) in el) or ((y,x) in el):
                    if random.random()<betweenGroupFactor:
                        validEdges.append((x,y))

    return validEdges

def getLoadableEL(validEdges,netInfo,CSEL,nRemove=0):
    net,n,k,p,r = netInfo["networkName"],netInfo["nSTN"],netInfo["k"],netInfo["p"],netInfo["r"]
    SG,GS,GG = get_SGGS_EL(validEdges,n)
    STG_list,GTS_list,GTG_list, graph_measures, all_normed_edges =  calc_network_measures(SG,GS,GG,n,dev=True)
    graphData = {"SGel":STG_list,"GSel":GTS_list,"GGel":GTG_list,"CSel":CSEL,
                 "graphMeasures":graph_measures,
                 "networkName":net,
                 "nSTN":n,"nGPe":n, "k":k,"p":p,"r":r,"converg":20}
    
    return graphData,graph_measures

def removeRandomNodes(fullEL,keep):
    indices = np.random.choice(range(len(fullEL)),size=keep,replace=False)
    el2 = [(fullEL[ind][0],fullEL[ind][1],fullEL[ind][2])  for ind in indices]
    return el2

def getLam2(eigs):
    lam2 = sorted(np.real(eigs))[1]
    return lam2


def runSim(netInfo,name,runTime = 1000,printOp=False,DBSparams = None):
    params = {
        "simtime": runTime,
        "h": 0.05,
        "STNbias": -1.0,
        "GPebias": -0.2,
        "SGdelay": 4,
        "GSdelay": 4,
        "GGdelay": 2,
        "Strweight": 0.25,
        "CTXweight": 0.25,
        "initPulseWidth": 15,
        "initPulseAmp": 2.5,
        "initPulseDuration": 300,
        "saveLFP": False,
                        }
    if DBSparams is None:
        DBSparams = {   "Enabled": False,
            "Amplitude": 60,
            "Frequency": 130,
            "PulseWidth": 0.2,
            "ChargeBalanceFactor": 10,
            "Start": 700,
            "End": 1300,
            "NodesToStim": [i for i in range(0)]}

    params["DBSparams"] = DBSparams
    params["networkInfo"] = netInfo
    loc = os.path.join("simRunData",f"{name}.npy")
    np.save(loc,params)
    COMMAND = ["python","newSim.py",name]
    subP = subprocess.Popen(COMMAND,stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = subP.communicate()
    if printOp:
        print(out.decode('UTF-8'))
        print(err.decode('UTF-8'))

def getUsefulRes(name):
    resultsLoc = os.path.join("Results",f"{name}_results.npy")
    results = np.load(resultsLoc,allow_pickle=True).item()
    syncDefault = results["Synchrony"]
    eigs = results["networkInfo"]["graphMeasures"]['eigs']
    return syncDefault,eigs

def get_SGGS_EL(all_edges,nSTN,deleteWeight=False):
    if deleteWeight:
        SG = [[int(a[0]), int(a[1]-nSTN)] for a in all_edges if ((a[0]<nSTN) and (a[1]>=nSTN))]
        GS = [[int(a[0]-nSTN), int(a[1])] for a in all_edges if ((a[0]>=nSTN) and (a[1]<nSTN))]
        GG = [[int(a[0]-nSTN), int(a[1]-nSTN)] for a in all_edges if ((a[0]>=nSTN) and (a[1]>=nSTN))]
    else:
        SG = [[int(a[0]), int(a[1]-nSTN), a[2]] for a in all_edges if ((a[0]<nSTN) and (a[1]>=nSTN))]
        GS = [[int(a[0]-nSTN), int(a[1]), a[2]] for a in all_edges if ((a[0]>=nSTN) and (a[1]<nSTN))]
        GG = [[int(a[0]-nSTN), int(a[1]-nSTN), a[2]] for a in all_edges if ((a[0]>=nSTN) and (a[1]>=nSTN))]

    return SG,GS,GG


def relabel_nodes(edgeList,CSEL):
    # Step 1: Generate a list of unique nodes from the given edge list.
    unique_nodes = set()
    for edge in edgeList:
        unique_nodes.add(edge[0])
        unique_nodes.add(edge[1])

    # Step 2: Create a dictionary mapping the original labels to new labels.
    labelMap = {old_label: new_label for new_label, old_label in enumerate(sorted(unique_nodes))}

    # Step 3: Replace each node in the edge list with its new label.
    relabeled_edge_list = [(labelMap[edge[0]], labelMap[edge[1]], edge[2]) for edge in edgeList]

    relabeledCSEL = [[e[0],labelMap[int(e[1])], e[2]]  for e in CSEL ]

    return relabeled_edge_list, relabeledCSEL,labelMap

def initlFromEdgelist(el,CTS,originalNetInfo,nSTN,nodeRemoval,reweight=True):
    #print(nSTN)
    SG,GS,GG = get_SGGS_EL(el,nSTN,deleteWeight=reweight)
    
    STG_list,GTS_list,GTG_list, graph_measures, all_normed_edges =  calc_network_measures(SG,GS,GG,nSTN,dev=True,reweight=reweight)

    if nodeRemoval:
        nNodes = len(set([e[0] for e in el] + [e[1] for e in el] )) 
        nGPe = nNodes - nSTN
        print(f"nSTN: {nSTN}, nGPe: {nGPe}")
    else:
        nGPe = originalNetInfo["nGPe"]

    
    netName = originalNetInfo["networkName"]
    k = originalNetInfo["k"]
    p = originalNetInfo["p"]
    r = originalNetInfo["r"]

    graphData = {"SGel":STG_list,"GSel":GTS_list,"GGel":GTG_list,"CSel":CTS,
                 "graphMeasures":graph_measures,
                 "networkName":netName,
                 "nSTN":nSTN,"nGPe":nGPe, "k":k,"p":p,"r":r,"converg":20,
                 "nCTX":originalNetInfo["nCTX"]}
    
    lsg = copy.deepcopy(STG_list)
    lgs = copy.deepcopy(GTS_list)
    lgg = copy.deepcopy(GTG_list)
    fullEL = update_index(lsg,0,nSTN,0) + update_index(lgs,nSTN,0,0) + update_index(lgg,nSTN,nSTN,0)

    return fullEL,graphData

def getCentralNodes2023(el):
    G = nx.Graph()
    G.add_edges_from([(e[0],e[1]) for e in el])
    centrality = nx.betweenness_centrality(G) #nx.eigenvector_centrality(G)
    mostCentral = [d[1] for d in sorted([(c,v) for v,c in centrality.items()],reverse=True)]
    return mostCentral

def getUsefulResBeta(name):
    resultsLoc = os.path.join("Results",f"{name}_results.npy")
    results = np.load(resultsLoc,allow_pickle=True).item()
    syncDefault = results["Synchrony"]
    eigs = results["networkInfo"]["graphMeasures"]['eigs']
    STNBeta = results["STNbeta"]
    GPeBeta = results["STNbeta"]
    SAMPA = results["SAMPA"]
    GAMPA = results["GAMPA"]
    SGABA = results["SGABA"]
    GGABA = results["GGABA"]
    return syncDefault,eigs,STNBeta,GPeBeta,results["SMean"],results["GMean"],SAMPA,GAMPA,SGABA,GGABA

# Function to calculate the sum of distances of non-zero elements from the diagonal
def diagonal_distance(row, row_idx,sum=False):
    col_indices = np.nonzero(row)[0]
    if sum:
        return np.sum(abs(col_indices - row_idx))
    else:
        return np.mean(abs(col_indices - row_idx)) #or sum

def cuthill_mckee(matrix):
    # Create a graph from the adjacency matrix
    G = nx.from_numpy_matrix(matrix)
    
    # Compute the Cuthill-McKee ordering of nodes
    r = nx.utils.reverse_cuthill_mckee_ordering(G)
    new_order = list(r)
    #maps old node numbers to new ones
    newToOld = {new:old for new,old in enumerate(new_order)}

    # Permute the rows and columns of the original adjacency matrix
    permuted_matrix = matrix.copy()
    permuted_matrix = permuted_matrix[new_order, :]
    permuted_matrix = permuted_matrix[:, new_order]
    
    return permuted_matrix, newToOld

def spectralReordering(el):
    #get eigenvectors and eigenvalues of normA
    eigs,vecs = np.linalg.eig(getNormalizedLaplacian(el))
    #sort the eigenvalues and get the list of sorting indices
    idx = eigs.argsort()
    lam2Index = idx[1]
    fielderVector = vecs[:,lam2Index]
    fieldIdx = fielderVector.argsort()
    
    newToOld = {new:old for new,old in enumerate(fieldIdx)}

    n = max([max(e) for e in el])+1
    defaultMat = np.zeros((n,n))
    for e in el:
        defaultMat[e[0],e[1]] = 1

    # Permute the rows and columns of the original adjacency matrix
    permuted_matrix = defaultMat.copy()
    permuted_matrix = permuted_matrix[fieldIdx, :]
    permuted_matrix = permuted_matrix[:, fieldIdx]
    
    return permuted_matrix, newToOld

def removeBestEdges(EL,nRemove,n,networkInfo,CSEL,reWeight):
    defaultMat = np.zeros((2*n,2*n))
    for e in EL:
        defaultMat[e[0],e[1]] = 1
    transformedMat, newToOld = cuthill_mckee(defaultMat)
    #transformedMat, newToOld = spectralReordering(EL)

    # get indices of non-zero elements
    rows, cols = np.nonzero(transformedMat)

    # convert indices to a list of tuples
    indices = list(zip(rows, cols))

    #sort by their distance from the diagonal
    indices.sort(key=lambda x: abs(x[0]-x[1]))

    #make a new list of the edges with the most distant edges removed
    newIndices = indices[:len(indices)-nRemove]

    #new matirx
    newMat = np.zeros((2*n,2*n))
    for e in newIndices:
        newMat[e[0],e[1]] = 1

    #transform the indices back to the original ordering
    newELA = [(newToOld[i[0]],newToOld[i[1]] ) for i in newIndices]

    newEL = []
    for e in EL:
        if (e[0],e[1]) in newELA:
            newEL.append(e)

    fullELPartition,networkInfoPartition =  initlFromEdgelist(newEL,CSEL,networkInfo,n,nodeRemoval=False,reweight=reWeight)
    eigs = np.real(networkInfoPartition["graphMeasures"]["eigs"])
    
    return sorted(eigs)[1],fullELPartition,networkInfoPartition


def removeBestNodes(EL,n,sum=False):
    #return getCentralNodes2023(EL)
    defaultMat = np.zeros((2*n,2*n))
    for e in EL:
        defaultMat[e[0],e[1]] = 1
    transformedMat, newToOld = cuthill_mckee(defaultMat)
    #transformedMat, newToOld = spectralReordering(EL)

    # Calculate distances for each row
    distances = np.array([diagonal_distance(row, i,sum=sum) for i, row in enumerate(transformedMat)])

    # Get the indices that would sort the distances array in descending order
    sorted_row_indices = np.argsort(-distances)
 
    return [newToOld[i] for i in sorted_row_indices]