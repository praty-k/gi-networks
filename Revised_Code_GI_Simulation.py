# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 22:34:15 2024

"""

import numpy as np
import networkx as nx
import EoN
#import h5py
import matplotlib.pyplot as plt
import time 
import sys


#%% Fxns
def sample_epidemic_ts(SimData, TimeBins):
    S, I, newS, newI = np.empty(len(TimeBins)), np.empty(len(TimeBins)), np.empty(len(TimeBins)), np.empty(len(TimeBins))
    newS[:] = np.nan
    newI[:] = np.nan
    Times = TimeBins-TimeBins[0]
    for counter, TimeBin in enumerate(TimeBins):
        State = np.array(list(SimData.get_statuses(time = TimeBin-TimeBins[0]).values()))
        S[counter] = np.sum(State == 'S')
        I[counter] = np.sum(State == 'I')
    PeakIndex = np.argmax(I)
    CentreIndex = int(len(TimeBins)/2)
    if CentreIndex > PeakIndex:
        newI[CentreIndex-PeakIndex:] = I[0:-(CentreIndex-PeakIndex)]
        newS[CentreIndex-PeakIndex:] = S[0:-(CentreIndex-PeakIndex)]
    elif CentreIndex < PeakIndex:
        newI[0:CentreIndex-PeakIndex] = I[-(CentreIndex-PeakIndex):]
        newS[0:CentreIndex-PeakIndex] = S[-(CentreIndex-PeakIndex):]
    else: 
        newI = I
        newS = S
    return newS, newI

def calculate_infection_events(SimData, PopSize, NumSeed):
    Transmissions = np.array(SimData.transmissions()) 
    InfectionTimes = -42.42*np.ones(PopSize)
    GenerationTimes = np.empty(len(Transmissions))
    InfectionTimes[Transmissions[:, 2].tolist()] = Transmissions[:, 0]
    GenerationTimes = Transmissions[NumSeed:, 0] - InfectionTimes[Transmissions[NumSeed:, 1].tolist()]
    GenerationTimes = np.array([GenerationTimes])
    InfectionTimes = np.array([InfectionTimes[Transmissions[NumSeed:, 1].tolist()]])
    InfectionEvents = np.hstack((Transmissions[NumSeed:], InfectionTimes.T, GenerationTimes.T))
    # [time of the infection event, infector, infectee, time of infection of infector, generation time of the event]
    return InfectionEvents 

def calculate_mean_generation_time(InfectionEvents, TimeBins):        
    #TimeBins = np.linspace(0, InfectionEvents[-1][0], NumBins)
    BackMeanGenTimes = []
    for RightEdge, LeftEdge in zip(TimeBins[1:], TimeBins[0:-1]):
        Events = InfectionEvents[(InfectionEvents[:, 0] > LeftEdge) & (InfectionEvents[:, 0] < RightEdge)]
        BackMeanGenTimes.append(np.mean(Events[:, -1]))
    FrontMeanGenTimes = []
    for RightEdge, LeftEdge in zip(TimeBins[1:], TimeBins[0:-1]):
        Events = InfectionEvents[(InfectionEvents[:, 3] > LeftEdge) & (InfectionEvents[:, 3] < RightEdge)]
        FrontMeanGenTimes.append(np.mean(Events[:, -1]))
    
    return TimeBins, BackMeanGenTimes, FrontMeanGenTimes

def calculate_forward_reproduction_number(InfectionEvents, TimeBins):        
    #TimeBins = np.linspace(0, InfectionEvents[-1][0], NumBins)    
    FrontR = []
    for i, Events in enumerate(InfectionEvents):
        if Events[3] == 0:
            InfectionEvents[i, 3] = -42
    for RightEdge, LeftEdge in zip(TimeBins[1:], TimeBins[0:-1]):
        Events = InfectionEvents[(InfectionEvents[:, 3] > LeftEdge) & (InfectionEvents[:, 3] < RightEdge)]
        NumInfections = len(Events)
        NumInfectors = len(InfectionEvents[(InfectionEvents[:, 0] > LeftEdge) & (InfectionEvents[:, 0] < RightEdge)])
        if NumInfectors == 0:
            FrontR.append(np.nan)
        else:
            FrontR.append(NumInfections/NumInfectors)
    
    return TimeBins, FrontR

#%%

if __name__ == '__main__':
    
    
    # Homogeneous Graph
    PopSize = 10000
    AvgDegree = 8
    DegreeSequence = [AvgDegree for _ in range(PopSize)]
    
    # Disease params
    beta = 0.5
    gamma = 1.0
    
    # Simulation params
    #NumIC = 10
    #NumNets = 1
    
    NumSeed = 1
    G = nx.configuration_model(DegreeSequence)
    G = nx.Graph(G)
    #G.remove_edges_from(nx.selfloop_edges(G))
    SimData = EoN.fast_SIR(G, beta, gamma, initial_infecteds = np.random.randint(0, PopSize, NumSeed), return_full_data=True)
    TimePeak = SimData.t()[np.argmax(SimData.I())]
    TimeMax = SimData.t()[-1]
    FinRec = SimData.R()[-1]
    FinSus = SimData.S()[-1]
    FinInf = SimData.I()[-1]
    
    InfectionEvents = calculate_infection_events(SimData, PopSize, NumSeed)
    TimeBins, BackMeanGenTimes, FrontMeanGenTimes = calculate_mean_generation_time(InfectionEvents, TimeBins = np.arange(0, 13, 0.5))
    
    #%% Power law like graph
    alpha = 2 #Exponent
    cutoff = 30 #Maximum degree in the network that is allowed
    degrees = np.array(list(range(1, cutoff+1))).astype(float)
    Nk = degrees**-alpha
    Nk = Nk/np.sum(Nk)
    
    PopSize = 10000
    
    DegreeSeq = np.random.choice(degrees, size = PopSize, p = Nk)
    
    # Disease params
    beta = 0.45
    gamma = 1.0
    
    # Simulation params
    #NumIC = 10
    #NumNets = 1
    
    NumSeed = 10
    G = nx.configuration_model(DegreeSequence)
    G = nx.Graph(G)
    #G.remove_edges_from(nx.selfloop_edges(G))
    SimData = EoN.fast_SIR(G, beta, gamma, initial_infecteds = np.random.randint(0, PopSize, NumSeed), return_full_data=True)
    TimePeak = SimData.t()[np.argmax(SimData.I())]
    TimeMax = SimData.t()[-1]
    FinRec = SimData.R()[-1]
    FinSus = SimData.S()[-1]
    FinInf = SimData.I()[-1]
    
    InfectionEvents = calculate_infection_events(SimData, PopSize, NumSeed)
    TimeBins, BackMeanGenTimes, FrontMeanGenTimes = calculate_mean_generation_time(InfectionEvents, NumBins = 40)

#%%
    plt.plot(TimeBins[1:], BackMeanGenTimes, lw = 0, marker = 'o', mec = 'k', mfc = 'none')
    plt.plot(TimeBins[0:-1], FrontMeanGenTimes, lw = 0, marker = '+', c = 'red')
    plt.ylim(0, 2)
