import neuron
h = neuron.h
from Cortical_Basal_Ganglia_Cell_Classes import GP_Neuron_Type, STN_Neuron_Type
import pyNN.neuron as sim
from pyNN.random import RandomDistribution as rnd
import numpy as np
import sys
import time as time
from datetime import datetime
from useful_functions_sonic import *
from additional_functions import *
import os

t0 = time.time()

paramFileName = sys.argv[1]
loc = os.path.join("simRunData",f"{paramFileName}.npy")
params = np.load(loc,allow_pickle=True).item()


nSTN = params["networkInfo"]["nSTN"]
nGPe = params["networkInfo"]["nGPe"]
nCTX = params["networkInfo"]["nCTX"]
#k = params["networkInfo"]["k"]
#p = params["networkInfo"]["p"]
SGel = params["networkInfo"]["SGel"]
GSel = params["networkInfo"]["GSel"]
GGel = params["networkInfo"]["GGel"]
CSel = params["networkInfo"]["CSel"]
converg = params["networkInfo"]["converg"]
simtime = params["simtime"]


### Set up the simulation
dt = params['h']           
max_syn_delay  = max(params['SGdelay'],params['GSdelay'],params['GGdelay'])
sim.setup(timestep=dt, max_delay=max_syn_delay)


### Create the neurons
v_init = rnd('uniform', (-66.0, -56.0))
a = STN_Neuron_Type(bias_current=params["STNbias"])
b = GP_Neuron_Type(bias_current=params["GPebias"])

STN_cells = sim.Population(nSTN, a ,initial_values={'v': v_init}, label="STN_cells")
GPe_cells = sim.Population(nGPe, b ,initial_values={'v': v_init}, label="GPe_cells")
CTX_Pop = sim.Population(nCTX,sim.SpikeSourcePoisson(rate=10) , label='Cortical Neuron Spike Source')
Striatal_Pop = sim.Population(converg*nGPe,sim.SpikeSourcePoisson(rate=3) , label='Striatal Neuron Spike Source')

### Establish connections
SG = sim.FromListConnector(SGel, column_names=["weight"])
GS = sim.FromListConnector(GSel, column_names=["weight"])
GG = sim.FromListConnector(GGel, column_names=["weight"])
CS = sim.FromListConnector(CSel, column_names=["weight"])
syn_STNGPe = sim.StaticSynapse( delay= params["SGdelay"])
syn_GPeSTN = sim.StaticSynapse( delay= params["GSdelay"])
syn_GPeGPe = sim.StaticSynapse( delay= params["GGdelay"])

    
syn_StriatalGPe = sim.StaticSynapse(weight=params['Strweight'] , delay=0)
syn_CTXSTN = sim.StaticSynapse(weight=params['CTXweight'] , delay=0)

prj_STNGPe = sim.Projection(STN_cells,GPe_cells, SG, syn_STNGPe, source='soma(0.5)', receptor_type='AMPA') #AMPA
prj_GPeSTN = sim.Projection(GPe_cells, STN_cells, GS, syn_GPeSTN, source='soma(0.5)', receptor_type='GABAa')
prj_GPeGPe = sim.Projection(GPe_cells,GPe_cells, GG, syn_GPeGPe, source='soma(0.5)', receptor_type='GABAa') #GABAa
prj_StriatalGPe = sim.Projection(Striatal_Pop, GPe_cells, sim.FixedNumberPreConnector(converg), 
                             syn_StriatalGPe, source='soma(0.5)', receptor_type='GABAa')
prj_CTXSTN = sim.Projection(CTX_Pop, STN_cells, CS, 
                             syn_CTXSTN, source='soma(0.5)', receptor_type='AMPA')

STNNoise = [sim.NoisyCurrentSource(mean=0, stdev = 0.05, start=0,stop =simtime,dt=1.0) for i in range(nSTN)] #was 20*dt which was usually .6ms
GPeNoise = [sim.NoisyCurrentSource(mean=0, stdev = 0.05, start=0,stop =simtime,dt=1.0) for i in range(nGPe)]


### Stimulation
# Desynchronizing initial pulse
npw= params["initPulseWidth"] #pulse width
namp= params["initPulseAmp"] #pulse amplitude
tStop = params["initPulseDuration"] #pulse duration
LFS = lambda x: [-1,1,0][x%3]
LFS_decider = lambda x,n: int(4*x/n)
LFS_stims = []
for toff in np.linspace(0,50,nSTN):
    x = sorted(np.arange(0+toff,simtime,50).tolist()+np.arange(0+toff+npw,simtime,50).tolist()+np.arange(0+toff+2*npw,simtime,50).tolist())
    y = [namp*LFS(i) if val<tStop else 0 for i,val in enumerate(x)]
    LFS_stims.append(sim.StepCurrentSource(times=x, amplitudes = y))
    

for i,cell in enumerate(STN_cells):
    cell.inject(STNNoise[i])
    #out of phase initialisation
    cell.inject(LFS_stims[i])

for i,cell in enumerate(GPe_cells):
    cell.inject(GPeNoise[i])


dt         = params['h']           # (ms)  best 0.03, use 0.1 for quick

sim.setup(timestep=dt, max_delay=max_syn_delay)

### Stimulation

cbf = params["DBSparams"]["ChargeBalanceFactor"]
stimstrt = params["DBSparams"]["Start"]
stimstop = params["DBSparams"]["End"]
pw = params["DBSparams"]["PulseWidth"]
amp = params["DBSparams"]["Amplitude"]
DBSperiod = 1000/params["DBSparams"]["Frequency"]


def cdb(x):
    if x%3==0:
        return 1
    elif x%3==1:
        return -1/cbf
    else:
        return 0
    
x = sorted(np.arange(stimstrt,stimstop,DBSperiod).tolist()+np.arange(stimstrt+pw,stimstop,DBSperiod).tolist()+np.arange(stimstrt+(pw*(1+cbf)),stimstop,DBSperiod).tolist())
y = [amp*cdb(i) for i in range(len(x))]
cs3 = sim.StepCurrentSource(times=x, amplitudes = y)

for i in params["DBSparams"]["NodesToStim"]:
    STN_cells[i].inject(cs3)


### Recording
STN_cells.record('spikes')
GPe_cells.record('spikes')
recordables = ['soma(0.5).v', 'AMPA.i','GABAa.i']
STN_cells.record(recordables)
GPe_cells.record(recordables)

### Run
sim.run(simtime)
t1 = time.time()

### Extract data
STN = STN_cells.get_data().segments[0]
GPe = GPe_cells.get_data().segments[0]

### Process data

data_dict = fill_dict(STN,GPe,dt,simtime)
if params["saveLFP"]:
    data_dict["STNdata"] = STN
    data_dict["GPedata"] = GPe


for key in params:
    data_dict[key] = params[key] 

now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
data_dict["When"] = now
data_dict["Time_taken"] = t1-t0
data_dict["nSG"] = len(SGel) 
data_dict["nGS"] = len(GSel)
data_dict["nGG"] = len(GGel)

data_dict["Synchrony"] = float(np.sqrt(data_dict["STN synchrony"]*data_dict["GPe synchrony"] ) )

SLFP = np.array(STN.filter(name='soma(0.5).v')[0]).mean(axis=1)
GLFP = np.array(GPe.filter(name='soma(0.5).v')[0]).mean(axis=1)
data_dict["STNbeta"] = getBetaPower(SLFP,dt)
data_dict["GPebeta"] = getBetaPower(GLFP,dt)

#update output dictionary with graph measures
dat = [data_dict]

outLoc = "Results"
outName = f"{paramFileName}_results.npy"
np.save(os.path.join(outLoc,outName),dat)