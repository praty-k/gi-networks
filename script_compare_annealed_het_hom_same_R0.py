# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:40:54 2024

"""


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Consolas"]

#%%
import cm_network_quenched_realised_gi as QE
import cm_network_annealed_realised_gi as MFE

#%% Keep R0 same for annealed and homogeneous and TPL
#%%% Transmission params
beta = 0.138
gamma = 1

#%%% TPL params
alpha = 2 #Exponent
cutoff = 70 #Maximum degree in the network that is allowed
degrees = np.array(list(range(1, cutoff+1))).astype(float)
Nk = degrees**-alpha
Nk = Nk/np.sum(Nk)

DegreeSquareAvg = np.sum(Nk*degrees**2)
DegreeAvg = np.sum(Nk*degrees)
HetR0 = beta/(gamma)*(DegreeSquareAvg)/DegreeAvg
print(DegreeAvg)
#%%% TPL Solve    
HetEpiMF = MFE.EpiSimNetMF(beta, gamma, degrees, Nk, max_t = 40, dt = 0.01, theta_0 = 1-1e-7)

TausMF = np.arange(0, 10, HetEpiMF.dt)
HetMFExactBackAvgs= np.empty(len(HetEpiMF.t))
HetMFExactFrontAvgs = np.empty(len(HetEpiMF.t))

for i in range(0, len(HetEpiMF.t)):
    HetMFExactBackAvgs[i] = MFE.mean_BGI_MF(i, TausMF, HetEpiMF.NewPi_I, beta, gamma)
    HetMFExactFrontAvgs[i] = MFE.mean_FGI_MF(i, TausMF, HetEpiMF.Pi_S, beta, gamma)
    


#%%% Homogeneous Solve    
beta_eff = HetR0/DegreeAvg
HomEpiMF = MFE.EpiSimNetMF(beta_eff, gamma, np.array([int(np.ceil(DegreeAvg))]), np.array([1]), max_t = 50, dt = 0.01, theta_0 = 1-1e-7)
HomR0 = beta_eff/(gamma)*DegreeAvg

TausMF = np.arange(0, 10, HomEpiMF.dt)
HomMFExactBackAvgs= np.empty(len(HomEpiMF.t))
HomMFExactFrontAvgs = np.empty(len(HomEpiMF.t))

for i in range(0, len(HomEpiMF.t)):
    HomMFExactBackAvgs[i] = MFE.mean_BGI_MF(i, TausMF, HomEpiMF.NewPi_I, beta_eff, gamma)
    HomMFExactFrontAvgs[i] = MFE.mean_FGI_MF(i, TausMF, HomEpiMF.Pi_S, beta_eff, gamma)

#%% Time shift 
'''-- set zero at prevalence = 0.01'''
HetEpiMF.ts = HetEpiMF.t - HetEpiMF.t[np.where(HetEpiMF.I >= 0.0001)[0][0]]
HomEpiMF.ts = HomEpiMF.t - HomEpiMF.t[np.where(HomEpiMF.I >= 0.0001)[0][0]]
#%% Plotting
# Het S, I
plt.figure(figsize=(7,9))
plt.suptitle('Annealed heterogeneous vs. homogeneous network: same $\\mathcal{R}_0$ and mean degree \n' + f'$\\beta_{{het}}$ = {beta}, $\\gamma$ = {gamma}, $\\beta_{{hom}}$ = {beta_eff:.2f}')

a1 = plt.subplot(3, 2, 1)
plt.plot(HetEpiMF.ts, HetEpiMF.NewPi_I, lw = 1, ls = '-', color = 'g', label = '$\pi_i(t)$')
plt.plot(HetEpiMF.ts, HetEpiMF.Pi_S, lw = 1, ls = '-', color = 'r', label = '$\pi_S(t)$')
plt.plot(HetEpiMF.ts, HetEpiMF.NewI, lw = 1, ls = '--', color = 'm', label = '$i(t)$')
plt.plot(HetEpiMF.ts, HetEpiMF.S, lw = 1, ls = '--', color = 'b', label = '$S(t)$')
plt.legend()
plt.xlim(0, 20)
plt.xlabel('$t$')
plt.title('Heterogeneous')

a2 = plt.subplot(3, 2, 2)
plt.plot(HomEpiMF.ts, HomEpiMF.NewPi_I, lw = 1, ls = '-', color = 'g', label = '$\pi_i(t)$')
plt.plot(HomEpiMF.ts, HomEpiMF.Pi_S, lw = 1, ls = '-', color = 'r', label = '$\pi_S(t)$')
plt.plot(HomEpiMF.ts, HomEpiMF.NewI, lw = 1, ls = '--', color = 'm', label = '$i(t)$')
plt.plot(HomEpiMF.ts, HomEpiMF.S, lw = 1, ls = '--', color = 'b', label = '$S(t)$')
plt.xlim(0, 20)
plt.legend()
plt.xlabel('$t$')
plt.title('Homogeneous')

# Het Back
plt.subplot(3, 2, 3, sharex = a1)
plt.plot(HetEpiMF.ts, HetMFExactBackAvgs, lw = 1, ls = '-', color = 'k', label = 'Back Avg')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0, 3)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')

plt.subplot(3, 2, 4, sharex = a2)
plt.plot(HomEpiMF.ts, HomMFExactBackAvgs, lw = 1, ls = '-', color = 'k', label = 'Back Avg')

plt.hlines(1/(gamma), HomEpiMF.ts[0], HomEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0, 3)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')
#plt.title('TPL degree distribution')

# Het Front
plt.subplot(3, 2, 5, sharex = a1)
plt.plot(HetEpiMF.ts, HetMFExactFrontAvgs, lw = 1, ls = '-', color = 'k', label = 'Front Avg')
plt.hlines(1/(gamma), HetEpiMF.ts[0], HetEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0.5*1/gamma, 1.2*1/gamma)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')

plt.subplot(3, 2, 6, sharex = a2)
plt.plot(HomEpiMF.ts, HomMFExactFrontAvgs, lw = 1, ls = '-', color = 'k', label = 'Front Avg')
plt.hlines(1/(gamma), HomEpiMF.ts[0], HomEpiMF.ts[-1], lw = 0.25, color = 'g')
plt.legend()
plt.ylim(0.5*1/gamma, 1.2*1/gamma)
plt.xlabel('$t$')
plt.ylabel(r'$\langle T \rangle$')
