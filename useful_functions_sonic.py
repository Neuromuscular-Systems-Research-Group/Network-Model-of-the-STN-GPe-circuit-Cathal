# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:25:24 2021

@author: catha
"""
import numpy as np
import networkx as nx

from scipy.spatial import distance_matrix 

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.signal as sg

from random import shuffle


#def meanCVspkfreq(trains):
#     '''Calculates the mean and coefficient of variation of the frequency for each neuron
#     In other words, the inverse of the interspike interval (ISI) 
#     and the CV of the inverse ISI
#     5/2/2021
#     '''
#     spikefreqs  = [1000/np.diff(np.array(i)) for i in trains ]
#     mean_freqs = [np.mean(i) for i in spikefreqs if not(len(i)==0)]
#     CV_freqs =  [np.std(i)/np.mean(i) for i in spikefreqs if not(len(i)==0)]
    
#     return mean_freqs,CV_freqs

def meanCVspkfreq(trains,simtime):
    '''Calculates firing rate by counting
    5/2/2021
    '''
    if simtime>1300:
        t2 = [ [j for j in i if j>1000] for i in trains ]
        mean_freqs = [(1000/(simtime-1000))*len(i) for i in t2]
    else:
        t2 = [ [j for j in i] for i in trains ]
        mean_freqs = [(1000/(simtime))*len(i) for i in t2]
        
    return mean_freqs


def bin_list(vals,bins=10):
    '''
    Returns histogram values and bin locations. You can then plot the histogram with plt.bar(b,c)
    Useful for performing chi squared test, and not much else 
    5/2/2021
    '''
    
    
    start = np.percentile(vals,5)
    step = (np.percentile(vals,95)-start)/10
    binvals=[start+step*i for i in range(bins)]
    print(binvals)
    a = np.histogram(vals, bins = binvals)
    b = a[1][:-1] + np.diff(a[1])/2
    c = a[0]
    return b,c


def medspkfreqs(trains):
    #spikefreqs  = [1000/np.diff(np.array(trains[i])) for i in range(n)]
    spikefreqs  = [1000/np.diff(np.array(i)) for i in trains]
    median_freqs = [np.median(i) for i in spikefreqs]
    return median_freqs

    

def gen_random_connec(n1,n2, k, allow_self_connec = False):
    rint = np.random.randint
    connections = []
    for i in range(k):
        for j in range(n1):
            proposition = (rint(n1),rint(n2))
            while (proposition in connections) or (proposition[0] == proposition[1]):
                proposition = (rint(n1),rint(n2))
            connections.append(proposition)
    return connections

def gen_ordered_connec_old(n1,n2, k, offset= 0, allow_self_connec = False):
    #will need to change this, wont work for clusters of different sizes. Should do it k in, instead of k out.
    connections = []
    for j in range(k):
        for i in range(n1):
            proposition = (i,(i+j+offset)%n2)
            connections.append(proposition)
    return connections

def gen_ordered_connec(n1,n2, k, connec_offset= 0, index_offset = 0, allow_self_connec = True):
    #will need to change this, wont work for clusters of different sizes. Should do it k in, instead of k out.
    connections = []
    if not(allow_self_connec):
        connec_offset += 1
    
    for j in range(k):
        for i in range(n1):
            proposition = (index_offset+i,index_offset+(i+j+connec_offset)%n2)
            connections.append(proposition)
    return connections


def gen_cluster_connect(n1,n2,k_per, k_inter, n_clusters,allow_self_connec = False):
    #ordered clusters
    connections = []
    
    n1_cluster_size = int(n1/n_clusters)
    n2_cluster_size = int(n2/n_clusters)
    n1_excess = n1 - n_clusters*n1_cluster_size
    n2_excess = n2 - n_clusters*n2_cluster_size
    
    max_connections = n1_cluster_size + int(allow_self_connec) -1
    
    if k_per>max_connections:
        raise Exception("use fewer intra cluster connections, max is {}".format(max_connections))
    
    
    
    if k_inter>n1_cluster_size:
        raise Exception("I cant deal with this right now, use fewer inter cluster connections")
    
    if not ((n1_excess == 0) and (n2_excess == 0)):
        raise Exception("Please ensure both populations are divisible by the number of clusters")
        
    for i in range(n_clusters):
            connections.extend(gen_ordered_connec(n1_cluster_size,n2_cluster_size, 
                                              k_per, connec_offset=0, index_offset = i*n1_cluster_size, allow_self_connec = allow_self_connec))
    
    for i in range(n_clusters):
        for j in range(k_inter):
            
            ind_x = (i+1)*n1_cluster_size-1-j
            ind_y = ((i+1)*n2_cluster_size+j)%n2
            
            connections.append( (ind_x,ind_y ) )
    
    return connections

def k_in(n1,n2,k, self_connec = False):
    connections = []
    n1_inds = [i for i in range(n1)]
    
    for i in range(n2):
        n1_inds = [p for p in range(n1)]
        
        if i<n1 and not(self_connec):
            n1_inds.remove(i)
        
        shuffle(n1_inds)
        proposition = [(n1_inds[j],i) for j in range(k) ]
        connections.extend(proposition)
    
    return connections


def k_out(n1,n2,k, self_connec = False):
    connections = []
    n1_inds = [i for i in range(n1)]
    
    for i in range(n2):
        n1_inds = [p for p in range(n1)]
        
        if i<n1 and not(self_connec): #check if self connections are allowed, if i>=n1 then theres simply no need to check
            n1_inds.remove(i) 
        
        shuffle(n1_inds) 
        proposition = [(i,n1_inds[j]) for j in range(k) ]
        connections.extend(proposition)
    
    return connections

def calc_reciprocity2(E):
    '''
    Calculates reciprocity as the ratio edges in the stronly connected graph to those in the weakly connected one.
    Similar to networkx function, but this one actually gives a value of 1 to a fully reciprocal graph
    '''
    C = E.to_undirected(reciprocal=False)
    D = E.to_undirected(reciprocal=True)
    s = D.number_of_edges()
    w = C.number_of_edges()

    return s/w


def drawnx(n1,n2,SG,GS,GG,GtS=1,StG=1,GtG=1,plot=1):
    B = nx.DiGraph()
    B.add_nodes_from(range(n1))
    B.add_nodes_from(range(n1,n1+n2))

    SG_nx = [[k[0],n1+k[1]] for k in SG]
    GS_nx = [[n1+k[0],k[1]] for k in GS]
    GG_nx = [[n1+k[0],n1+k[1]] for k in GG]

    if StG:
        B.add_edges_from(SG_nx)
    if GtS:
        B.add_edges_from(GS_nx)
    if GtG:
        B.add_edges_from(GG_nx)

    #res = {i: [int(i/n),i%n] for i in range(n1+n2)}
    
    print("Reciprocity: {:.2f}%".format( 100*calc_reciprocity2(B)))
    
    # x_pos = [0 if i<n1 else 0.5+0.1*(i%4) for i in range(n1+n2)]
    # y_pos = [i if i<n2 else i-n1 for i in range(n1+n2)]
    
    # res = {i: [x_pos[i],y_pos[i]] for i in range(n1+n2)}
    
    
    # if plot:
    #     nx.draw(B, pos = res)
    #     plt.title("Network Plot")

    #     offsetx = 0.02
    #     offsety = 2
    #     plt.text(-offsetx+0,-offsety+0, "STN",fontsize=16)
    #     plt.text(-offsetx+1,-offsety+0, "GPe",fontsize=16)

    return B


def get_nearest_index(line):
    x = [i[1] for i in line]
    ind = x.index(min(x))
    return line[ind][0]

def removeind(row,l,k):
    for i in l:
        if i[0]==k:
            row.remove(i)

def removecond(row,l,k):
    for i in l:
        if i[1]==k:
            row.remove(i)

def add_near_edge(edgelist,roww,j):
    near_ind = get_nearest_index(roww)
    edgelist.append([j,near_ind])
    for i in roww:
        if i[0]==near_ind:
            roww.remove(i)
    return near_ind

def remove_next_nearest(near_ind,distmat,roww, num_times=1):
    y_row =  [[i,val] for i,val in enumerate(distmat[near_ind])]
    for i in y_row:
        if i[1]==0.0:
            #distyy is has zeros along diagonal so need to remove these as options (closest node is itself, doesnt make sense here)
            y_row.remove(i)
        
    
    for i in range(num_times):
        near_indy = get_nearest_index(y_row) #find closest node in y to the one we have chosen
        for i in roww:
            if i[0]==near_indy:
                roww.remove(i)

        for i in y_row:
            if i[0]==near_indy:
                y_row.remove(i)
            

      
def gen_nnn_edges_xy(n,k,a,b,sparsity = 1):
    
    distxy = distance_matrix(a,b,p=2) #distances from points x to points y
    distyy = distance_matrix(b,b,p=2) #distances between points y
    edges = []

    for node_id in range(n):
        row = [[i,val] for i,val in enumerate(distxy[node_id])]
        if type(k)==list:
            no_connec = np.random.randint(k[0],k[1])
        else:
            no_connec = k
        for i in range(no_connec):
            relevant_ind = add_near_edge(edges,row,node_id)
            remove_next_nearest(relevant_ind,distyy,row,num_times=sparsity)
            
    return edges


def bin_degs(de):
    a = np.histogram(de, bins = [12+4*i for i in range(11)])
    b = a[1][:-1] + np.diff(a[1])/2
    c = a[0]
    return b,c

    

def get_domdata(dat,a,n):
    di = {"Freq":0,
         "Power":1,
          "LK":2,
          "SK":3
         }
    y = np.zeros([n,n,n])
    for i in range(n):
        for j in range(n):
            for k in range(n):
                y[i,j,k] = dat[i*n + j*n + k][di[a]]
                     
    if a=="Freq":
        y[y>150] = 0
        
    return y 

def make_cheby1_filter(Fs, N, rp, low, high):
	"""Calculate bandpass filter coefficients (1st Order Chebyshev Filter)"""
	nyq = 0.5*Fs
	lowcut = low/nyq
	highcut = high/nyq

	b, a = sg.cheby1(N, rp, [lowcut, highcut], 'band')
	#b, a = sg.butter(N, [lowcut, highcut], 'band')
    
	return b, a
	
def calculate_avg_power(lfp_signal, beta_b, beta_a):
    """Calculate the average power in the beta-band for the current LFP signal window, i.e. beta Average Rectified Value (ARV)

    Exaqmple inputs:
        lfp_signal 			- window of LFP signal												# samples
        tail_length 		- tail length which will be discarded due to filtering artifact		# samples
        beta_b, beta_a 		- filter coefficients for filtering the beta-band from the signal	
    """
    lfp_beta_signal = sg.filtfilt(beta_b, beta_a, lfp_signal)
    lfp_beta_signal_rectified = np.absolute(lfp_beta_signal)
    
    avg_beta_power = np.mean(lfp_beta_signal_rectified)

    return avg_beta_power


def get_peak_freq(LFP, dt):
    X = abs(np.fft.fft(LFP-np.mean(LFP)))
    freq = np.linspace(0,1000/dt,len(X))

    Y = sg.savgol_filter(X, 2*int(1/dt)+1, 3, deriv = 0, mode = 'nearest')
    Y = Y[0:int(1000/dt)]

    S_Peak_freq = freq[np.nonzero([Y==np.max(Y)])[1][0]]
    S_Peak_val = np.max(Y)
    return S_Peak_freq,S_Peak_val