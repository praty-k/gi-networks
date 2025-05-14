# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:37:58 2024

@author: PKollepara
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Consolas"]
# Ground Reality, 
#Population 1
alpha = 2 #Exponent
cutoff = 70 #Maximum degree in the network that is allowed
degrees = np.array(list(range(1, cutoff+1))).astype(float)
Nk = degrees**-alpha
Nk = Nk/np.sum(Nk)

DegreeSquareAvg = np.sum(Nk*degrees**2)
DegreeAvg = np.sum(Nk*degrees)
k1 = DegreeAvg
kr1 = DegreeSquareAvg/DegreeAvg

lambda1 = 0.5
Gamma = 1.0
beta1 = (lambda1 + Gamma)/(kr1-2)
T1_exp = 1/(lambda1 + beta1 + Gamma)



# Population 2
alpha = 2 #Exponent
cutoff = 30 #Maximum degree in the network that is allowed
degrees = np.array(list(range(1, cutoff+1))).astype(float)
Nk = degrees**-alpha
Nk = Nk/np.sum(Nk)

DegreeSquareAvg = np.sum(Nk*degrees**2)
DegreeAvg = np.sum(Nk*degrees)
k2 = DegreeAvg

kr2 = DegreeSquareAvg/DegreeAvg

lambda2s = np.arange(0, 2.1, 0.1)
Gamma = 1.0
beta2s = (lambda2s + Gamma)/(kr2-2)
R0_2s = 1 + lambda2s/(beta2s + Gamma)

# Case A:
EstimatorR0_1 = 1/(1-lambda1*T1_exp)
EstimatorR0_2s = (1 + (lambda2s - lambda1)*T1_exp)/(1-lambda1*T1_exp)
EstimatorGamma = 1/T1_exp - lambda1

plt.figure(figsize=(7,4))
plt.suptitle(f'Bias in $\Re_0$ estimate for Population 2')
a1 = plt.subplot(1, 2, 1)
plt.plot(lambda2s, EstimatorR0_2s, label = 'Estimate', lw = 0.1, marker = 'd', color = 'm', markersize = 3)
plt.plot(lambda2s, R0_2s, label = 'True', lw = 0.5, marker = '.', color = 'k', ls = ':')
plt.legend()
plt.xlabel('Growth rate in Population 2')
plt.title('Well mixed homogeneous model')

# Case B: 
EstimatorR0_1 = 1/(1-lambda1*T1_exp)
EstimatorGamma = ((k1-2)*(1/T1_exp - lambda1)-lambda1)/(k1-1)
EstimatorBeta2 = (lambda2s + EstimatorGamma)/(k2-2)
EstimatorR0_2s = 1 + lambda2s/(EstimatorBeta2 + EstimatorGamma)

plt.subplot(1, 2, 2, sharex = a1, sharey = a1)
plt.plot(lambda2s, EstimatorR0_2s, label = 'Estimate', lw = 0.1, marker = 'd', color = 'm', markersize = 3)
plt.plot(lambda2s, R0_2s, label = 'True', lw = 0.5, marker = '.', color = 'k', ls = ':')
plt.legend()
plt.xlabel('Growth rate in Population 2')
plt.title('Quenched homogeneous model')
plt.tight_layout()
#plt.savefig('Manuscript-Figures/R0_estimate')