from additional_functions import *
import subprocess 
import os

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


net = "ImprovedSpatial"
n,k,p = 100,10,0.1
duration = 1000 #ms
printOP = True

fullEL,networkInfo = initialiseNet(net,n,k,p,r=1.0)
defaultName = f"{net}_{n}_{k}_{p}_DEFAULT"
runSim(networkInfo,defaultName,runTime=duration,printOp=printOP)
