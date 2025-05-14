# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:58:50 2024

@author: pkollepara
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Consolas"]
#%%
import cm_network_quenched_realised_gi as QE
import cm_network_annealed_realised_gi as MFE

'''Annelaed het and hom, same peak'''
#%% Transmission params
beta = 0.138
gamma = 1

#%%% TPL params
net_type = 'TPL70'

if net_type == 'TMD':
    ''' Tri modal degree dist '''
    degrees = np.array([3, 5, 7])
    Nk = np.array([1/3, 1/3, 1/3])
    beta = .542
    t_max = 15
elif net_type == 'MMD':
    degrees = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17])
    Nk = np.ones(len(degrees))*1/len(degrees)
    beta = .542/2
    t_max = 12
elif net_type == 'Uniform':
    degrees = np.arange(1, 18, 1)
    Nk = np.ones(len(degrees))*1/len(degrees)
    beta = .542/2
    t_max = 20
elif net_type == 'TPL70':
    ''' TPL70 Network '''
    alpha = 2 #Exponent
    cutoff = 70 #Maximum degree in the network that is allowed
    degrees = np.array(list(range(1, cutoff+1))).astype(float)
    Nk = degrees**-alpha
    Nk = Nk/np.sum(Nk)
    beta = 0.138
    t_max = 40


DegreeSquareAvg = np.sum(Nk*degrees**2)
DegreeAvg = np.sum(Nk*degrees)
HetR0 = beta/(gamma)*(DegreeSquareAvg)/DegreeAvg
print(DegreeAvg)
#%%% TPL Solve    
HetEpiMF = MFE.EpiSimNetMF(beta, gamma, degrees, Nk, max_t = t_max, dt = 0.01, theta_0 = 1-1e-7)

TausMF = np.arange(0, t_max, HetEpiMF.dt)
HetMFExactBackAvgs= np.empty(len(HetEpiMF.t))
HetMFExactFrontAvgs = np.empty(len(HetEpiMF.t))

for i in range(0, len(HetEpiMF.t)):
    HetMFExactBackAvgs[i] = MFE.mean_BGI_MF(i, TausMF, HetEpiMF.NewPi_I, beta, gamma)
    HetMFExactFrontAvgs[i] = MFE.mean_FGI_MF(i, TausMF, HetEpiMF.Pi_S, beta, gamma)

HetMFExactBackAvgsFromNodes= np.empty(len(HetEpiMF.t))
HetMFExactFrontAvgsFromNodes = np.empty(len(HetEpiMF.t))

for i in range(0, len(HetEpiMF.t)):
    HetMFExactBackAvgsFromNodes[i] = MFE.mean_BGI_MF(i, TausMF, HetEpiMF.NewI, beta, gamma)
    HetMFExactFrontAvgsFromNodes[i] = MFE.mean_FGI_MF(i, TausMF, HetEpiMF.S, beta, gamma)

#%% Time shift 
'''-- set zero at prevalence = 0.01'''
HetEpiMF.ts = HetEpiMF.t - HetEpiMF.t[np.where(HetEpiMF.I >= 0.0001)[0][0]]
HetEpiMF.ts = HetEpiMF.t
#%% Plotting
# Het S, I
plt.figure(figsize=(7,9))
plt.suptitle('Annealed heterogeneous network: stubs vs. nodes \n' + f'$\\beta_{{het}}$ = {beta}, $\\gamma$ = {gamma}')

a1 = plt.subplot(3, 2, 1)


plt.plot(HetEpiMF.ts, HetEpiMF.NewPi_I, lw = 1, ls = '-', color = 'g', label = '$\pi_i(t)$')
plt.plot(HetEpiMF.ts, HetEpiMF.Pi_S, lw = 1, ls = '-', color = 'r', label = '$\pi_S(t)$')
#plt.plot(HetEpiMF.ts, HetEpiMF.NewI, lw = 1, ls = '--', color = 'm', label = '$i(t)$')
#plt.plot(HetEpiMF.ts, HetEpiMF.S, lw = 1, ls = '--', color = 'b', label = '$S(t)$')
plt.legend()
plt.xlim(0, 20)
plt.xlabel('$t$')
plt.title('From stubs (correct)')

a2 = plt.subplot(3, 2, 2, sharey = a1)
plt.plot(HetEpiMF.ts, HetEpiMF.NewI, lw = 1, ls = '--', color = 'm', label = '$i(t)$')
plt.plot(HetEpiMF.ts, HetEpiMF.S, lw = 1, ls = '--', color = 'b', label = '$S(t)$')
plt.xlim(0, 20)
plt.legend()
plt.xlabel('$t$')
plt.title('From nodes (incorrect)')

# Het Back
plt.subplot(3, 2, 3, sharex = a1)
plt.plot(HetEpiMF.ts, HetMFExactBackAvgs, lw = 1, ls = '-', color = 'k', label = 'Back Avg (from stubs)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0, 3)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')

plt.subplot(3, 2, 4, sharex = a2)
plt.plot(HetEpiMF.ts, HetMFExactBackAvgsFromNodes, lw = 1, ls = '-', color = 'k', label = 'Back Avg \n(from nodes, incorrect)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0, 3)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')
#plt.title('TPL degree distribution')

# Het Front
plt.subplot(3, 2, 5, sharex = a1)
plt.plot(HetEpiMF.ts, HetMFExactFrontAvgs, lw = 1, ls = '-', color = 'k', label = 'Front Avg (from stubs)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0.5*1/gamma, 1.2*1/gamma)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')

plt.subplot(3, 2, 6, sharex = a2)
plt.plot(HetEpiMF.ts, HetMFExactFrontAvgsFromNodes, lw = 1, ls = '-', color = 'k', label = 'Front Avg \n(from stubs, incorrect)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0.5*1/gamma, 1.2*1/gamma)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')

plt.tight_layout()
#plt.savefig('het_hom_same_peak_prev.png', dpi = 600)

#%%

plt.figure(figsize=(7,9))
plt.suptitle('Annealed heterogeneous network: stubs vs. nodes \n' + f'$\\beta_{{het}}$ = {beta}, $\\gamma$ = {gamma}')

a1 = plt.subplot(3, 1, 1)


plt.plot(HetEpiMF.ts, HetEpiMF.NewPi_I, lw = 1, ls = '-', color = 'g', label = '$\pi_i(t)$')
plt.plot(HetEpiMF.ts, HetEpiMF.Pi_S, lw = 1, ls = '-', color = 'r', label = '$\pi_S(t)$')
#plt.plot(HetEpiMF.ts, HetEpiMF.NewI, lw = 1, ls = '--', color = 'm', label = '$i(t)$')
#plt.plot(HetEpiMF.ts, HetEpiMF.S, lw = 1, ls = '--', color = 'b', label = '$S(t)$')
plt.legend()
plt.xlim(0, 20)
plt.xlabel('$t$')
plt.title('From stubs (correct)')


plt.plot(HetEpiMF.ts, HetEpiMF.NewI, lw = 1, ls = '--', color = 'm', label = '$i(t)$')
plt.plot(HetEpiMF.ts, HetEpiMF.S, lw = 1, ls = '--', color = 'b', label = '$S(t)$')
plt.xlim(0, 20)
plt.legend()
plt.xlabel('$t$')
plt.title('From nodes (incorrect)')

# Het Back
plt.subplot(3, 1, 2, sharex = a1)
plt.plot(HetEpiMF.ts, HetMFExactBackAvgs, lw = 1, ls = '-', color = 'k', label = 'Back Avg (from stubs)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')

plt.plot(HetEpiMF.ts, HetMFExactBackAvgsFromNodes, lw = 1, ls = '-', color = 'k', label = 'Back Avg \n(from nodes, incorrect)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0, 2)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')
#plt.title('TPL degree distribution')

# Het Front
plt.subplot(3, 1, 3, sharex = a1)
plt.plot(HetEpiMF.ts, HetMFExactFrontAvgs, lw = 1, ls = '-', color = 'k', label = 'Front Avg (from stubs)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')

plt.plot(HetEpiMF.ts, HetMFExactFrontAvgsFromNodes, lw = 1, ls = '-', color = 'k', label = 'Front Avg \n(from stubs, incorrect)')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0.5*1/gamma, 1.2*1/gamma)
plt.ylim(0.8, 1.1)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')

plt.tight_layout()
#plt.savefig('het_hom_same_peak_prev.png', dpi = 600)
