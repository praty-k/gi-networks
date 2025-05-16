# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:07:35 2024

"""

import numpy as np
import EoN
import networkx as nx
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Consolas"]
#%%
import cm_network_quenched_realised_gi as exact
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

#%%
gamma = 1    

#%%% Exact
net_type = 'TMD' #'TMD'

if net_type == 'TPL70':
    ''' TPL70 Network '''
    alpha = 2 #Exponent
    cutoff = 70 #Maximum degree in the network that is allowed
    degrees = np.array(list(range(1, cutoff+1))).astype(float)
    Nk = degrees**-alpha
    Nk = Nk/np.sum(Nk)
    #beta = 0.25 #TPL70
    beta = 0.286 #R0 = 3
    
    
elif net_type == 'TMD':
    ''' Tri modal degree dist '''
    degrees = np.array([3, 5, 7])
    Nk = np.array([1/3, 1/3, 1/3])
    beta = 1.956 #TMD
    #TimeBins = np.arange(0, 24, 0.5)
    #ZeroBin = np.argmin(np.abs(TimeBins))
    

elif net_type == 'UMD':
    ''' Uni modal degree dist '''
    degrees = np.array([3])
    Nk = np.array([1])
    beta = 10
    

DegreeSquareAvg = np.sum(Nk*degrees**2)
DegreeAvg = np.sum(Nk*degrees)

HetR0 = beta/(beta+gamma)*(DegreeSquareAvg-DegreeAvg)/DegreeAvg
HetEpi = exact.EpiSimNetQ(beta, gamma, degrees, Nk, max_t = 20, dt = 0.01,)
#T, S, I, R, Phi_S, Phi_I, Phi_R, NewPhi_I, theta = [HetEpi[key] for key in HetEpi.keys()]

Taus = np.arange(0, 10, HetEpi.dt)
HetExactBackAvgs, HetExactBackUL, HetExactBackLL = np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t))
HetExactFrontAvgs, HetExactFrontUL, HetExactFrontLL = np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t)), np.empty(len(HetEpi.t))

# for i in range(0, len(HetEpi.t)):
#     HetExactBackAvgs[i], HetExactBackUL[i], HetExactBackLL[i] = exact.mean_BGI_Q(i, Taus, HetEpi.NewPhi_I, beta, gamma)
#     HetExactFrontAvgs[i], HetExactFrontUL[i], HetExactFrontLL[i] = exact.mean_FGI_Q(i, Taus, HetEpi.Phi_S, beta, gamma)
HetExactBackAvgs = [exact.mean_BGI_Q(i, Taus, HetEpi.NewPhi_I, beta, gamma) for i in range(0, len(HetEpi.t))]
HetExactFrontAvgs = [exact.mean_FGI_Q(i, Taus, HetEpi.Phi_S, beta, gamma) for i in range(0, len(HetEpi.t))] # Forward generation average
HetExactFRN = [exact.FRN_Q(i, Taus, HetEpi.Phi_S, HetEpi.theta, Nk, degrees, beta, gamma) for i in range(0, len(HetEpi.t))] # FRN



#%%% Simulation
NumSims = 100
PopSize = 10000
TimeBins = np.arange(-10, 10, 0.25)
DegreeSequence = np.random.choice(degrees.astype(int), size = PopSize, p = Nk)
if np.sum(DegreeSequence)%2 != 0:
    DegreeSequence[0] +=1 
G = nx.configuration_model(DegreeSequence)
G = nx.Graph(G)

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
    HetSimData = EoN.fast_SIR(G, beta, gamma, initial_infecteds = np.random.randint(0, PopSize, NumSeed), return_full_data=True)
    if HetSimData.t()[-1] < 1/gamma*6: #Four average generation times
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
plt.suptitle(f'Quenched Network, Topology: {net_type}\n'+f'$\\beta$ = {beta}, $\\gamma$ = {gamma}, $\\mathcal{{R}}_0$ = {HetR0:.2f}')
a1 = plt.subplot(4, 1, 1)
plt.plot(TimeBins[::2], AvgHetSimData.I[::2], lw = 0, mec = 'k', mfc = 'none', marker = '.', label = 'I (sim)')
plt.plot(TimeBins, Is/PopSize, lw = .1, color = 'r', alpha = 0.5)
plt.plot(HetEpi.t, HetEpi.I, lw = 1, ls = '--', color = 'm', label = 'I (exact)')

plt.plot(TimeBins[::2], AvgHetSimData.S[::2], lw = 0, mec = 'k', mfc = 'none', marker = 's', label = r'S (sim)')
plt.plot(TimeBins, Ss/PopSize, lw = .1, color = 'gray', alpha = 0.5)
plt.plot(HetEpi.t, HetEpi.S, lw = 1, ls = '-.', color = 'b', label = 'S (exact)')

plt.legend()
plt.ylim(0, 1)
plt.xlim(-3, 3)

# Het Back
plt.subplot(4, 1, 2, sharex = a1)
plt.plot(TimeBins[1:], HetSimBacks, lw = 0.1, color = 'thistle')
plt.plot(HetEpi.t, HetExactBackAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.plot(TimeBins[1:], HetSimBackAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.hlines(1/(gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0, 2)
plt.title('Mean backward generation interval')

# Het Front
plt.subplot(4, 1, 3, sharex = a1)
plt.plot(TimeBins[1:], HetSimFronts, lw = 0.1, color = 'thistle')
plt.plot(HetEpi.t, HetExactFrontAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.plot(TimeBins[0:-1], HetSimFrontAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.hlines(1/(gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0, 1.1)
plt.title('Mean forward generation interval')

plt.subplot(4, 1, 4, sharex = a1)
plt.plot(TimeBins[0:-1], HetSimFRNs, lw = 0.1, color = 'thistle')
#plt.plot(HetEpi.t, HetExactFrontAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact Front Avg')
plt.plot(TimeBins[0:-1], HetSimFRNAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Simulation')
plt.plot(HetEpi.t, HetExactFRN, lw = 1, ls = '--', color = 'k', label = 'Exact')
plt.legend()
plt.title(r'Forward reproduction number $\mathcal{R}_f$')
plt.xlabel('$t$')
plt.ylim(0, 7)
plt.tight_layout()
plt.savefig(f'Manuscript-Figures/Quenched_exact_sim_{net_type}.png', dpi = 300)


# Hom S, I
# plt.subplot(3, 2, 2)
# plt.plot(HomEpi.t, HomEpi.I, lw = 1, ls = '--', color = 'm', label = 'Exact, I')
# plt.plot(TimeBins, AvgHomSimData.I/PopSize, lw = 0, mec = 'r', mfc = 'none', marker = '.', label = 'Sim, I')
# plt.plot(HomEpi.t, HomEpi.S, lw = 1, ls = '--', color = 'b', label = 'Exact, S')
# plt.plot(TimeBins, AvgHomSimData.S/PopSize, lw = 0, color = 'k', mfc = 'none', marker = '.', label = 'Sim, I')
# plt.legend()
# plt.ylim(0, 1)
# plt.title('Homogeneous degree distribution')

# # Hom Back
# plt.subplot(3, 2, 4)
# plt.plot(TimeBins[1:], HomSimBacks, lw = 0.1, color = 'gray')
# plt.plot(HomEpi.t, HomExactBackAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact Back Avg')
# plt.plot(TimeBins[1:], HomSimBackAvg, lw = 0, mec = 'm', mfc = 'none', marker = '.', label = 'Sim Back Avg')
# plt.hlines(1/(beta+gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.25, color = 'g')
# plt.legend()
# plt.ylim(0, 3)
# #plt.title('Homogeneous degree distribution')

# # Hom Front
# plt.subplot(3, 2, 6)
# plt.plot(TimeBins[1:], HomSimFronts, lw = 0.1, color = 'gray')
# plt.plot(HomEpi.t, HomExactFrontAvgs, lw = 1, ls = '--', color = 'k', label = 'Exact Front Avg')
# plt.plot(TimeBins[0:-1], HomSimFrontAvg, lw = 0, mec = 'b', mfc = 'none', marker = '+', label = 'Sim Front Avg')
# plt.hlines(1/(beta+gamma), HetEpi.t[0], HetEpi.t[-1], lw = 0.25, color = 'g')
# plt.legend()
# plt.ylim(0, 1.5)
#plt.title('Homogeneous degree distribution')


#%% Homogeneous network
#%%% Exact
# k = int(np.ceil(DegreeAvg))
# degrees = np.array([k])
# Nk = np.array([1])
# DegreeSquareAvg = np.sum(Nk*degrees**2)
# DegreeAvg = np.sum(Nk*degrees)
# beta = gamma*HetR0/(k - 1 - HetR0)
# HomR0 = beta/(beta+gamma)*(DegreeSquareAvg-DegreeAvg)/DegreeAvg
# HomEpi = exact.EpiSimNetQ(beta, gamma, degrees, Nk, max_t = 20, dt = 0.01,)
# #T, S, I, R, Phi_S, Phi_I, Phi_R, NewPhi_I, theta = [HomEpi[key] for key in HomEpi.keys()]

# Taus = np.arange(0, 10, HomEpi.dt)
# HomExactBackAvgs, HomExactBackUL, HomExactBackLL = np.empty(len(HomEpi.t)), np.empty(len(HomEpi.t)), np.empty(len(HomEpi.t))
# HomExactFrontAvgs, HomExactFrontUL, HomExactFrontLL = np.empty(len(HomEpi.t)), np.empty(len(HomEpi.t)), np.empty(len(HomEpi.t))
# # for i in range(0, len(HomEpi.t)):
# #     HomExactBackAvgs[i], HomExactBackUL[i], HomExactBackLL[i] = exact.mean_BGI_Q(i, Taus, HomEpi.NewPhi_I, beta, gamma)
# #     HomExactFrontAvgs[i], HomExactFrontUL[i], HomExactFrontLL[i] = exact.mean_FGI_Q(i, Taus, HomEpi.Phi_S, beta, gamma)
# HomExactBackAvgs = [exact.mean_BGI_Q(i, Taus, HomEpi.NewPhi_I, beta, gamma) for i in range(0, len(HomEpi.t))]
# HomExactFrontAvgs = [exact.mean_FGI_Q(i, Taus, HomEpi.Phi_S, beta, gamma) for i in range(0, len(HomEpi.t))] # Forward generation average

# #%%% Simulation
# NumSims = 100
# PopSize = 10000
# DegreeSequence = np.random.choice(degrees.astype(int), size = PopSize, p = Nk)
# if np.sum(DegreeSequence)%2 != 0:
#     DegreeSequence[0] +=1 
# G = nx.configuration_model(DegreeSequence)
# G = nx.Graph(G)

# NumSeed = 8
# HomSimBacks = np.zeros((len(TimeBins)-1, NumSims))
# HomSimFronts = np.zeros((len(TimeBins)-1, NumSims))
# AvgHomSimData = AvgEON(TimeBins, np.zeros(len(TimeBins)), np.zeros(len(TimeBins)), np.zeros(len(TimeBins)))
# for counter in range(NumSims):
#     HomSimData = EoN.fast_SIR(G, beta, gamma, initial_infecteds = np.random.randint(0, PopSize, NumSeed), return_full_data=True)
#     InfectionEvents = simulator.calculate_infection_events(HomSimData, PopSize, NumSeed)
#     TimeBins, x, y = simulator.calculate_mean_generation_time(InfectionEvents, TimeBins)
#     HomSimBacks[:, counter] = x
#     HomSimFronts[:, counter] = y
    
#     S, I, R = EoN.subsample(TimeBins, HomSimData.t(), HomSimData.S(), HomSimData.I(), HomSimData.R())
#     AvgHomSimData.S += S/NumSims
#     AvgHomSimData.I += I/NumSims
#     AvgHomSimData.R += R/NumSims
#     #print(np.isnan(x).sum())
# HomSimBackAvg = np.nanmean(HomSimBacks, axis = 1)
# HomSimFrontAvg = np.nanmean(HomSimFronts, axis = 1)


#%%% Show all simulation curves


# Het Back
#plt.plot(TimeBins[1:], HetSimBacks, lw = 0, mec = 'gray', mfc = 'none', marker = '_', alpha = 0.1)


# Het Front
#plt.plot(TimeBins[0:-1], HetSimFronts, lw = 0, mec = 'gray', mfc = 'none', marker = '_', alpha = 0.1)


# Hom Front
#plt.plot(TimeBins[0:-1], HomSimFronts, lw = 0, mec = 'gray', mfc = 'none', marker = '_', alpha = 0.1)
