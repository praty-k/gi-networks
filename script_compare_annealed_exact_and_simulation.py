# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 13:31:45 2025

"""

# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:07:35 2024

@author: pkollepara
"""

import numpy as np
import EoN
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import random
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Consolas"]
#%% Functions
import cm_network_annealed_realised_gi as exact
import Revised_Code_GI_Simulation as simulator

class AvgEON(object):
    def __init__(self, t, S, I, R):
        self.t = t
        self.S = S
        self.I = I
        self.R = R

def NaN(a1, a2=1):
    if a2 == 1:
        x = np.zeros(a1)
    else:
        x = np.zeros((a1, a2))
    x[:] = np.nan
    return x

def centre_on_time(InfectionEvents, time_centre):
    InfectionEvents[:, 0] = InfectionEvents[:, 0] - time_centre
    InfectionEvents[:, 3] = InfectionEvents[:, 3] - time_centre
    return InfectionEvents

def subsample(Data, time_centre, TimeBins):
    times = Data.t() - time_centre
    S = NaN(len(TimeBins))
    I = NaN(len(TimeBins))
    for counter, Time in enumerate(TimeBins):
        if Time + time_centre >= 0:
            states = np.array(list(Data.get_statuses(time = Time + time_centre).values()))
            S[counter] = np.sum(states == 'S')
            I[counter] = np.sum(states == 'I')
    return S, I

def Gillespie_HMF(degreedict, tau, gamma, initial_infecteds=None, 
                    initial_recovereds = None, rho = None, tmin = 0, 
                    tmax=float('Inf'), 
                    return_full_data = False, sim_kwargs = None):

    #degreedict should have a node's name as key and its degree as 'dict'.  
    #
    nodelist = list(degreedict.keys())

    if rho is not None and initial_infecteds is not None:
        raise EoN.EoNError("cannot define both initial_infecteds and rho")
    
    if return_full_data:
        infection_times = defaultdict(lambda: []) #defaults to an empty list for each node
        recovery_times = defaultdict(lambda: [])
    tau = float(tau)  #just to avoid integer division problems in python 2.
    gamma = float(gamma)
    if initial_infecteds is None:
        if rho is None:
            initial_number = 1
        else:
            initial_number = int(round(len(nodelist)*rho))
        initial_infecteds=random.sample(nodelist, initial_number) #should really test that these aren't in intitial_recovereds
    elif initial_infecteds in nodelist:  #in case they give just a single node.
        initial_infecteds=[initial_infecteds]

    if initial_recovereds is None:
        initial_recovereds = []

    if not set(initial_infecteds).isdisjoint(initial_recovereds):  #the two sets overlap
        raise EoN.EoNError("initial_infecteds and initial_recovereds overlap")
    if not set(initial_infecteds).issubset(nodelist):
        raise EoN.EoNError("initial_infecteds has nodes not in population")
    if not set(initial_recovereds).issubset(nodelist):
        raise EoN.EoNError("initial_recovereds has nodes not in population")

    infecteds_for_infection = EoN.simulation._ListDict_(weighted=True) #weighted list
    infecteds_for_recovery = EoN.simulation._ListDict_(weighted=False) #unweighted list
    fullpopulation = EoN.simulation._ListDict_(weighted=True)

    I = [len(initial_infecteds)]
    R = [len(initial_recovereds)]
    S = [len(nodelist)-I[0]-R[0]]

    times = [tmin]
    transmissions = []
    t = tmin

    status = defaultdict(lambda : 'S')
    for node in initial_infecteds:
        status[node] = 'I'
        if return_full_data:
            infection_times[node].append(t)
            transmissions.append((t, None, node))
    for node in initial_recovereds:
        status[node] = 'R'
        if return_full_data:
            recovery_times[node].append(t)
    for node in nodelist:
        fullpopulation.update(node, weight_increment = degreedict[node])
        if status[node] == 'I':
            infecteds_for_infection.update(node, weight_increment = degreedict[node])
            infecteds_for_recovery.update(node)

    node = None #just to make sure I don't accidentally use it later.


    #inf_kave = infecteds_for_infection.total_weight()/len(infecteds_for_infection)
    #print('initial infected kave = ', inf_kave)

    #kave_of_infecteds = [inf_kave]
    total_recovery_rate = gamma*I[-1]
    total_transmission_rate = tau*infecteds_for_infection.total_weight()

    total_rate = total_recovery_rate + total_transmission_rate

    delay = random.expovariate(total_rate)
    t += delay

    while infecteds_for_infection and t<tmax:
        #assert(I[-1] == infecteds_for_infection.__len__())
        #assert(I[-1] == len(infecteds_for_recovery))
        if random.random()<total_recovery_rate/total_rate: #recover 
            recovering_node = infecteds_for_recovery.random_removal() #does random choice and removes it
            infecteds_for_infection.remove(recovering_node)
            status[recovering_node]='R'
            if return_full_data:
                recovery_times[recovering_node].append(t)
            times.append(t)
            S.append(S[-1])
            I.append(I[-1]-1)
            R.append(R[-1]+1)
            #if I[-1]>0:
            #    kave_of_infecteds.append(infecteds_for_infection.total_weight()/len(infecteds_for_infection))   
            #else:
            #    kave_of_infecteds.append(0)
        else: #transmit
            transmitter = infecteds_for_infection.choose_random()
            recipient = fullpopulation.choose_random()  #would be good to update this to weight it so we just choose from susceptibles, but for now...
            if status[recipient] == 'S':  #an infection happens
                if return_full_data:
                    transmissions.append((t, transmitter, recipient))
                    infection_times[recipient].append(t)
                    
                infecteds_for_infection.update(recipient, weight_increment = degreedict[recipient])
                infecteds_for_recovery.update(recipient)
                status[recipient] = 'I'
                times.append(t)
                S.append(S[-1] - 1)
                I.append(I[-1]+1)
                R.append(R[-1])
                #kave_of_infecteds.append(infecteds_for_infection.total_weight()/len(infecteds_for_infection))   
        total_recovery_rate = gamma*I[-1]
        total_transmission_rate = tau*infecteds_for_infection.total_weight()
        total_rate = total_recovery_rate + total_transmission_rate
        
        if total_rate>0:
            delay = random.expovariate(total_rate)
        else:
            delay = float('Inf')
        t += delay  #note - if nothing happens, we still advance time, but don't record

    if not return_full_data:
        return np.array(times), np.array(S), np.array(I), \
                np.array(R)#, np.array(kave_of_infecteds)
    else:
        infection_times = {node: L[0] for node, L in infection_times.items()}
        recovery_times = {node: L[0] for node, L in recovery_times.items()}
        G = nx.Graph()
        G.add_nodes_from(nodelist)  #just an empty graph
        node_history = EoN.simulation._transform_to_node_history_(infection_times, recovery_times, tmin, SIR = True)
        if sim_kwargs is None:
            sim_kwargs = {}
        return EoN.Simulation_Investigation(G, node_history, transmissions, possible_statuses = ['S', 'I', 'R'], **sim_kwargs)

#%%
gamma = 1    

#%%% Exact
net_type = 'TPL70'

if net_type == 'TPL70':
    ''' TPL70 Network '''
    alpha = 2 #Exponent
    cutoff = 70 #Maximum degree in the network that is allowed
    degrees = np.array(list(range(1, cutoff+1))).astype(float)
    Nk = degrees**-alpha
    Nk = Nk/np.sum(Nk)
    #beta = 0.25 #TPL70
    beta = 0.207 #Ensures R0 = 3
    
    
elif net_type == 'TMD':
    ''' Tri modal degree dist '''
    degrees = np.array([3, 5, 7])
    Nk = np.array([1/3, 1/3, 1/3])
    #beta = 0.5 #TMD
    beta = 0.542 #Ensures R0 = 3
    #TimeBins = np.arange(0, 24, 0.5)
    #ZeroBin = np.argmin(np.abs(TimeBins))
    

elif net_type == 'UMD':
    ''' Uni modal degree dist '''
    degrees = np.array([3])
    Nk = np.array([1])
    beta = .75
    

DegreeSquareAvg = np.sum(Nk*degrees**2)
DegreeAvg = np.sum(Nk*degrees)

HetR0 = beta/gamma*(DegreeSquareAvg)/DegreeAvg
HetEpi = exact.EpiSimNetMF(beta, gamma, degrees, Nk, max_t = 20, dt = 0.01,)
#T, S, I, R, Phi_S, Phi_I, Phi_R, NewPhi_I, theta = [HetEpi[key] for key in HetEpi.keys()]

Taus = np.arange(0, 10, HetEpi.dt)
#HetExactBackAvgs, HetExactBackUL, HetExactBackLL = np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t))
#HetExactFrontAvgs, HetExactFrontUL, HetExactFrontLL = np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t))

# for i in range(0, len(HetEpi.t)):
#     HetExactBackAvgs[i], HetExactBackUL[i], HetExactBackLL[i] = exact.mean_BGI_Q(i, Taus, HetEpi.NewPhi_I, beta, gamma)
#     HetExactFrontAvgs[i], HetExactFrontUL[i], HetExactFrontLL[i] = exact.mean_FGI_Q(i, Taus, HetEpi.Phi_S, beta, gamma)
HetExactBackAvgs = [exact.mean_BGI_MF(i, Taus, HetEpi.NewPi_I, beta, gamma) for i in range(0, len(HetEpi.t))]
HetExactFrontAvgs = [exact.mean_FGI_MF(i, Taus, HetEpi.Pi_S, beta, gamma) for i in range(0, len(HetEpi.t))] # Forward generation average
HetExactFRN = [exact.FRN_MF(i, Taus, HetEpi.Pi_S, HetEpi.theta, Nk, degrees, beta, gamma) for i in range(0, len(HetEpi.t))] # FRN



#%%% Simulation
NumSims = 500
PopSize = 10000
TimeBins = np.arange(-10, 10, 0.25)
DegreeSequence = np.random.choice(degrees.astype(int), size = PopSize, p = Nk)
if np.sum(DegreeSequence)%2 != 0:
    DegreeSequence[0] +=1 
G = nx.configuration_model(DegreeSequence)
G = nx.Graph(G)
degreedict = dict(G.degree)
NumSeed = 3
#HetSimBacks = np.zeros((len(TimeBins)-1, NumSims))
#HetSimFronts = np.zeros((len(TimeBins)-1, NumSims))
#HetSimFRNs = np.zeros((len(TimeBins)-1, NumSims))
HetSimBacks = NaN(len(TimeBins)-1, NumSims)
HetSimFronts = NaN(len(TimeBins)-1, NumSims)
HetSimFRNs = NaN(len(TimeBins)-1, NumSims)
Ss = NaN(len(TimeBins), NumSims)
Is = NaN(len(TimeBins), NumSims)
Rs = NaN(len(TimeBins), NumSims)

AvgHetSimData = AvgEON(TimeBins, np.zeros(len(TimeBins)), np.zeros(len(TimeBins)), np.zeros(len(TimeBins)))
SuccessSims = 0
for counter in range(NumSims):
    HetSimData = Gillespie_HMF(degreedict, beta, gamma, initial_infecteds = np.random.randint(0, PopSize, NumSeed).tolist(), return_full_data=True)
    if HetSimData.t()[-1] < 1/gamma*4: 
        continue
    SuccessSims += 1
    TimePeak = HetSimData.t()[np.argmax(HetSimData.I())]
    InfectionEvents = simulator.calculate_infection_events(HetSimData, PopSize, NumSeed)
    InfectionEvents = centre_on_time(InfectionEvents, TimePeak)
    TimeBins, x, y = simulator.calculate_mean_generation_time(InfectionEvents, TimeBins)
    TimeBins, FrontR = simulator.calculate_forward_reproduction_number(InfectionEvents, TimeBins)
    HetSimBacks[:, counter] = x
    HetSimFronts[:, counter] = y
    HetSimFRNs[:, counter] = FrontR
    Ss[:, counter], Is[:, counter] = subsample(HetSimData, TimePeak, TimeBins)
    #Ss[:, counter], Is[:, counter], Rs[:, counter] = EoN.subsample(TimeBins - TimeBins[0], HetSimData.t(), HetSimData.S(), HetSimData.I(), HetSimData.R())
    #print(np.isnan(x).sum())
AvgHetSimData.t = TimeBins
AvgHetSimData.S = np.nanmean(Ss, axis = 1)/PopSize
AvgHetSimData.I = np.nanmean(Is, axis = 1)/PopSize
AvgHetSimData.R = np.nanmean(Rs, axis = 1)/PopSize

HetSimBackAvg = np.nanmean(HetSimBacks, axis = 1)
HetSimFrontAvg = np.nanmean(HetSimFronts, axis = 1)
HetSimFRNAvg = np.nanmean(HetSimFRNs, axis = 1)



#%%%
# Het S, I
'''-- set zero at starting prevalence of simulation'''
#OT = HetEpi.t
#HetEpi.t = HetEpi.t - HetEpi.t[np.argmin(np.abs(HetEpi.I - NumSeed/PopSize))]
#HomEpiQ.ts = HomEpiQ.t - HomEpiQ.t[np.argmax(HomEpiQ.I)]

'''-- set zero at peak prevalence'''
OT = HetEpi.t
HetEpi.t = HetEpi.t - HetEpi.t[np.argmax(HetEpi.I)]
#HomEpiQ.ts = HomEpiQ.t - HomEpiQ.t[np.argmax(HomEpiQ.

#%%
plt.figure(figsize=(4,9))
plt.suptitle(f'Annealed Network, Topology: {net_type}\n'+f'$\\beta$ = {beta}, $\\gamma$ = {gamma}, $\\Re_0$ = {HetR0:.2f}')
a1 = plt.subplot(4, 1, 1)
plt.plot(TimeBins[::2], AvgHetSimData.I[::2], lw = 0, mec = 'k', mfc = 'none', marker = '.', label = 'I (sim)')
plt.plot(TimeBins, Is/PopSize, lw = .1, color = 'r', alpha = 0.5)
plt.plot(HetEpi.t, HetEpi.I, lw = 1, ls = '--', color = 'm', label = 'I (exact)')

plt.plot(TimeBins[::2], AvgHetSimData.S[::2], lw = 0, mec = 'k', mfc = 'none', marker = 's', label = r'S (sim)')
plt.plot(TimeBins, Ss/PopSize, lw = .1, color = 'gray', alpha = 0.5)
plt.plot(HetEpi.t, HetEpi.S, lw = 1, ls = '-.', color = 'b', label = 'S (exact)')

plt.legend()
plt.ylim(0, 1)
plt.xlim(-5, 4)

# Het Back
plt.subplot(4, 1, 2, sharex = a1)
plt.plot(TimeBins[1:], HetSimBacks, lw = 0.1, color = 'thistle')
plt.plot(HetEpi.t, HetExactBackAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.plot(TimeBins[1:], HetSimBackAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.hlines(1/(gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.5, color = 'g')
plt.legend()
plt.ylim(0, 2)
plt.title('Mean backward generation interval')

# Het Front
plt.subplot(4, 1, 3, sharex = a1)
plt.plot(TimeBins[1:], HetSimFronts, lw = 0.1, color = 'thistle')
plt.plot(HetEpi.t, HetExactFrontAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.plot(TimeBins[0:-1], HetSimFrontAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.hlines(1/(gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.5, color = 'g')
plt.legend()
plt.ylim(0, 1.5)
plt.title('Mean forward generation interval')

plt.subplot(4, 1, 4, sharex = a1)
plt.plot(TimeBins[0:-1], HetSimFRNs, lw = 0.1, color = 'thistle')
plt.plot(TimeBins[0:-1], HetSimFRNAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.plot(HetEpi.t, HetExactFRN, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.legend()
plt.title(r'Forward reproduction number $\Re_f$')
plt.xlabel('$t$')
plt.ylim(0, 7)
plt.tight_layout()

#plt.savefig(f'Manuscript-Figures/Annealed_exact_sim_{net_type}.png', dpi = 300)

#%% Figure for ppt. Demo of what FGI and BGI mean look like

# 1 time unit = 3 days
Dt = 4
HetExactBackAvgs = np.array(HetExactBackAvgs)
HetExactFrontAvgs = np.array(HetExactFrontAvgs)

plt.figure(figsize=(4,5))
plt.suptitle(f'Mean intrinsic generation time = {1/gamma*Dt} days')
a1 = plt.subplot(3, 1, 1)
plt.plot(HetEpi.t*Dt, HetEpi.I, lw = 1, ls = '--', color = 'm', label = 'I(t)')
plt.plot(HetEpi.t*Dt, HetEpi.S, lw = 1, ls = '-.', color = 'b', label = 'S(t)')

plt.legend()
plt.ylim(0, 1)
plt.xlim(0, 14*Dt)

# Het Back
plt.subplot(3, 1, 2, sharex = a1)

plt.plot(HetEpi.t*Dt, HetExactBackAvgs*Dt, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.hlines(1/(gamma)*Dt, HetEpi.t[0], HetEpi.t[-1]*Dt, lw = .7, color = 'g')
#plt.legend()
plt.ylim(0, 2.5*Dt)
plt.title('Mean backward generation interval')

# Het Front
plt.subplot(3, 1, 3, sharex = a1)
plt.plot(HetEpi.t*Dt, HetExactFrontAvgs*Dt, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.hlines(1/(gamma)*Dt, HetEpi.t[0], HetEpi.t[-1]*Dt, lw = .7, color = 'g')
#plt.legend()
plt.xlabel(r'$t$ (days)')
plt.ylim(0, 1.2*Dt)
plt.title('Mean forward generation interval')

plt.tight_layout()
#plt.savefig('Talk-Figures/UMD FGI BGI.png', dpi = 300, transparent=True)

#%%Figure for ppt. FGI, BGI and S, I 

plt.figure(figsize=(4,5))
plt.suptitle(f'Annealed Network, Topology: {net_type}\n'+f'$\\gamma$ = {gamma}, $\\Re_0$ = {HetR0:.2f}')
a1 = plt.subplot(3, 1, 1)
plt.plot(TimeBins, Is/PopSize, lw = .1, color = 'salmon', alpha = 0.5)
plt.plot(TimeBins[::2], AvgHetSimData.I[::2], lw = 0, mec = 'k', mfc = 'none', marker = '.', label = 'I (sim)')
plt.plot(HetEpi.t, HetEpi.I, lw = 1, ls = '--', color = 'm', label = 'I (exact)')
plt.title('Prevalence')

#plt.plot(TimeBins[::2], AvgHetSimData.S[::2], lw = 0, mec = 'k', mfc = 'none', marker = 's', label = r'S (sim)')
#plt.plot(TimeBins, Ss/PopSize, lw = .1, color = 'gray', alpha = 0.5)
#plt.plot(HetEpi.t, HetEpi.S, lw = 1, ls = '-.', color = 'b', label = 'S (exact)')

#plt.legend()
plt.ylim(0, )
plt.xlim(-5, 4)

# Het Back
plt.subplot(3, 1, 2, sharex = a1)
plt.plot(TimeBins[1:], HetSimBacks, lw = 0.1, color = 'thistle')
plt.plot(HetEpi.t, HetExactBackAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.plot(TimeBins[1:], HetSimBackAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.hlines(1/(gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.5, color = 'g')
#plt.legend()
plt.ylim(0, 2)
plt.title('Mean backward generation interval')

# Het Front
plt.subplot(3, 1, 3, sharex = a1)
plt.plot(TimeBins[1:], HetSimFronts, lw = 0.1, color = 'thistle')
plt.plot(HetEpi.t, HetExactFrontAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.plot(TimeBins[0:-1], HetSimFrontAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.hlines(1/(gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.5, color = 'g')
#plt.legend()
plt.ylim(0.5, 1.5)
plt.title('Mean forward generation interval')

plt.xlabel('$t$')
plt.tight_layout()
#plt.savefig(f'Talk-Figures/{net_type} Compare.png', dpi = 300, transparent=True)
