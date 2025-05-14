# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 14:29:38 2024

@author: pkollepara
"""

# Package importation
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% Object solution def
class MeanFieldSolution(object):
    def __init__(self, beta, gamma, degrees, Nk, max_t, dt, t, S, I, R, Pi_S, Pi_I, Pi_R, NewPi_I, NewI, theta):
        self.beta = beta
        self.gamma = gamma
        self.degrees = degrees
        self.Nk = Nk
        self.max_t = max_t
        self.dt = dt
        self.t = t
        self.S = S
        self.I = I
        self.R = R
        self.Pi_S = Pi_S
        self.Pi_I = Pi_I
        self.Pi_R = Pi_R
        self.NewPi_I = NewPi_I
        self.NewI = NewI
        self.theta = theta

#%% Fxn definitions
def Psi(x, Nk, deg):
    n = len(deg)
    return np.sum([Nk[i]*x**deg[i] for i in range(n)], 0)
    #return np.sum(Nk*x**(np.array(degrees)))

def PsiPrime(x, Nk, deg):
    n = len(deg)
    return np.sum([Nk[i]*deg[i]*x**(deg[i]-1) for i in range(n)], 0)

def PsiDoublePrime(x, Nk, deg):
    n = len(deg)
    return np.sum([Nk[i]*deg[i]*(deg[i]-1)*x**(deg[i]-2) for i in range(n)], 0)

def dxdt(t, X, rho, beta, gamma, Nk, degrees):
    R, Pi_R, theta = X
    Sk_0 = Nk*(1-rho)
    S = np.sum(Sk_0*theta**degrees)
    Pi_S = np.sum(degrees*Nk*theta**degrees)/PsiPrime(1.0, Nk, degrees)
    Pi_I = 1 - Pi_S - Pi_R
    I = 1 - S - R
    return np.array([gamma*I, gamma*Pi_I, -beta*Pi_I*theta])

def FS(x, G, n): 
    return [x[k] - n[k]*(1-np.exp(-np.sum(G[k]*x)/n[k])) for k in range(len(n))]

# Fxn for mean backward generatio interval
def mean_BGI_MF(t_index, Taus, rate_new_infections, beta, gamma, draw = False):
    N = len(Taus)
    dt = Taus[1] - Taus[0]
    g_Taus = gamma*np.exp(-gamma*Taus)
    """denominator = np.sum([dt*p_Taus[i]*rate_new_infections[t_index - i] for i in range(0, N)])"""
    Rates = []
    for i in range(0, N):
        if t_index - i >= 0:
            Rates.append(g_Taus[i]*rate_new_infections[t_index - i])
        else:
            Rates.append(0)
    
    RatesSum = np.sum(Rates)*dt
    
    b_Taus = Rates/RatesSum
    
    TauAvg = np.sum(b_Taus*Taus*dt)
    
    if draw:
        plt.plot(Taus, b_Taus, label = 'BGI distribution')
        plt.plot(Taus, g_Taus, label = 'IGI distribution')
        plt.title(f'{t_index = }')
        plt.legend()
        return (TauAvg, Rates, RatesSum, b_Taus)
    else:
        return TauAvg

#Fxn for mean forward generation intervals
def mean_FGI_MF(t_index, Taus, Pi_S, beta, gamma, draw = False):
    N = len(Taus)
    dt = Taus[1] - Taus[0]
    g_Taus = gamma*np.exp(-gamma*Taus)
    """denominator = np.sum([dt*p_Taus[i]*rate_new_infections[t_index - i] for i in range(0, N)])"""
    Rates = []
    for i in range(0, N):
        if t_index + i < len(Pi_S):
            Rates.append(g_Taus[i]*Pi_S[t_index + i])
        else:
            Rates.append(0)
    
    RatesSum = np.sum(Rates)*dt
    
    f_Taus = Rates/RatesSum
    
    TauAvg = np.sum(f_Taus*Taus*dt)
    
    if draw:
        plt.plot(Taus, f_Taus, label = 'FGI distribution')
        plt.plot(Taus, g_Taus, label = 'IGI distribution')
        plt.title(f'{t_index = }')
        plt.legend()
        return (TauAvg, Rates, RatesSum, f_Taus)
    else:
        return TauAvg

#Fxn for forward reproduction number
def FRN_MF(t_index, Taus, Pi_S, theta, Nk, degrees, beta, gamma):
    N = len(Taus)
    dt = Taus[1] - Taus[0]
    p_Taus = np.exp(-(gamma)*Taus)
    """denominator = np.sum([dt*p_Taus[i]*rate_new_infections[t_index - i] for i in range(0, N)])"""
    Rates = []
    for i in range(0, N):
        if t_index + i < len(Pi_S):
            Rates.append(p_Taus[i]*Pi_S[t_index + i])
        else:
            Rates.append(0)
    
    RatesSum = np.sum(Rates)*dt
    f_Taus = Rates/RatesSum
    R_f = beta*(PsiDoublePrime(theta[t_index], Nk, degrees)*theta[t_index] + PsiPrime(theta[t_index], Nk, degrees))\
               /PsiPrime(theta[t_index], Nk, degrees)*RatesSum
    
    return R_f

def EpiSimNetMF(beta, gamma, degrees, Nk, max_t, dt, theta_0 = 1-1e-4, method = 'Radau'):
    T = np.arange(0, max_t, dt)
    #theta_0 = 1-1e-5
    X_0 = [0, 0, theta_0]
    Sk_0 = Nk*theta_0**degrees
    rho_k = 1 - theta_0**degrees
    n = len(Nk)
    sol = solve_ivp(lambda t, X: dxdt(t, X, rho_k, beta, gamma, Nk, degrees), t_span=(0, T[-1]), y0 = X_0, method = method, t_eval = T[0:-1])
    R, Pi_R, theta = sol.y[0, :], sol.y[1, :], sol.y[2, :]
    S = np.sum(np.array([Nk[i]*theta**degrees[i] for i in range(n)]), 0)        
    Psioftheta = Psi(theta, Nk, degrees)
    I = (1-S-R)
    #S_approx = Psioftheta*np.exp(-Chi)
    Pi_S = theta*PsiPrime(theta, Nk, degrees)/PsiPrime(1.0, Nk, degrees)
    Pi_I = 1 - Pi_S -Pi_R
    NewPi_I = beta*Pi_I*(theta**2*PsiDoublePrime(theta, Nk, degrees) + theta*PsiPrime(theta, Nk, degrees))/(PsiPrime(1, Nk, degrees))
    
    NewI = beta*Pi_I*theta*PsiPrime(theta, Nk, degrees)
    epidemic = MeanFieldSolution(beta, gamma, degrees, Nk, max_t, dt, sol.t, S, I, R, Pi_S, Pi_I, Pi_R, NewPi_I, NewI, theta)
    return epidemic


if __name__ == '__main__':
    #%% Heterogeneous MF SIR on 
    
    #Power law
    alpha = 2 #Exponent
    cutoff = 30 #Maximum degree in the network that is allowed
    degrees = np.array(list(range(1, cutoff+1))).astype(float)
    Nk = degrees**-alpha
    Nk = Nk/np.sum(Nk)
    
    # degrees = np.array([3, 5, 7])
    # Nk = np.array([1/3, 1/3, 1/3])
    beta = 2
    beta_net = beta*np.sum(degrees*Nk)/np.sum(Nk*degrees**2)
    #beta_net = 0.5
    gamma = 1 
    
    HetEpi = EpiSimNetMF(beta_net, gamma, degrees, Nk, 40, 0.01,)
    T, S, I, R, Pi_S, Pi_I, Pi_R, NewI, NewPi_I, theta = [HetEpi[key] for key in HetEpi.keys()]
    
    #plt.plot(T, NewPi_I, lw = 0.5)
    #plt.legend()
    #NewPi_I = Pi_I*(theta**2*PsiDoublePrime(theta, Nk, degrees) + theta*PsiPrime(theta, Nk, degrees))/(PsiPrime(1, Nk, degrees))
    # plt.plot(T, I, label = 'I')
    # plt.plot(T, NewI, label = 'NewI')
    # plt.plot(T, NewPi_I, label = 'NewPi_I')
    # plt.hlines(0, 0, 40, lw = 0.5, colors = 'black')
    # plt.legend()
    #plt.close()
    #%%
    Taus = np.arange(0, 10, T[1]-T[0])
    
    TauBackAvgs = [mean_BGI_MF(i, Taus, HetEpi['NewI'], beta_net, gamma) for i in range(0, len(T))]
    
    TauBackAvgsStubs = [mean_BGI_MF(i, Taus, NewPi_I, beta_net, gamma) for i in range(0, len(T))]
    
    TauFrontAvgs = [mean_FGI_MF(i, Taus, S, beta_net, gamma) for i in range(0, len(T))] # Forward generation average
    
    TauFrontAvgsStubs = [mean_FGI_MF(i, Taus, Pi_S, beta_net, gamma) for i in range(0, len(T))]
    
    
    #mean_BGI_MF_from_incidence(1749, Taus, i_rate, beta, gamma, draw = True)
    
    #%%
    f, axs = plt.subplots(3, 1, sharex = True)
    plt.suptitle(f'Power-law like network. k_max = {cutoff}. $\\beta$ = {beta_net:.2f}. $\\gamma$ = {gamma}')
    axs[0].plot(T, NewI, label = '$i$')
    axs[0].plot(T, NewPi_I, label = '$\pi_I$')
    axs[0].plot(T, S, label = '$S$')
    axs[0].plot(T, Pi_S, label = '$\pi_S$')
    #axs[0].set_yscale('log')
    axs[0].legend()
    axs[1].plot(T, TauBackAvgs, label = 'from incidence of infections')
    axs[1].plot(T, TauBackAvgsStubs, label = 'from incidence of stubs')
    axs[1].set(title = 'Mean back generation interval')
    axs[1].legend()
    axs[2].plot(T, TauFrontAvgs, label = 'FGI: from S')
    axs[2].plot(T, TauFrontAvgsStubs, label = 'FGI: from S-stubs')
    axs[2].set_ylim(0.5, )
    axs[2].set(title = 'Mean forward generation interval')
    axs[2].legend()
    axs[0].set_xlim(0, 25)
    axs[0].set_ylim(0, )
    axs[1].set_ylim(0, 2)
    
    #%% Homogeneous model with effective parameters to ensure same R0 and growth rate
    
    HomEpi = EpiSimNetMF(beta, gamma, np.array([1]), np.array([1]), 40, 0.01,)
    TauBackHomAvgs = [mean_BGI_MF(i, Taus, HomEpi['NewPi_I'], beta, gamma) for i in range(0, len(T))]
    TauFrontHomAvgs = [mean_FGI_MF(i, Taus, HomEpi['S'], beta, gamma) for i in range(0, len(T))]
    
    #%% Power law vs homogenous plot
    f, axs = plt.subplots(3, 1, sharex = True)
    plt.suptitle(f'Power-law like network vs effective homogeneous model.\n' + 
                 f'k_max = {cutoff}. $\\beta$ = {beta_net:.2f}. $\\gamma$ = {gamma}. ' + 
                 f'$\\beta_{{eff}}$ = {beta:.2f}')
    axs[0].plot(T, I, label = 'Het $I$')
    axs[0].plot(T, S, label = 'Het $S$')
    axs[0].plot(HomEpi['t'], HomEpi['I'], label = 'Hom $I$', ls = '--')
    axs[0].plot(HomEpi['t'], HomEpi['S'], label = 'Hom $S$', ls = '--')
    #axs[0].set_yscale('log')
    axs[0].legend()
    
    axs[1].plot(T, TauBackAvgsStubs, label = r'Het $\langle \tau \rangle_{back}$')
    axs[1].plot(HomEpi['t'], TauBackHomAvgs, label = r'Hom $\langle \tau \rangle_{back}$', ls = '--')
    axs[1].set(title = 'Mean back generation interval')
    axs[1].legend()
    
    axs[2].plot(T, TauFrontAvgsStubs, label = r'Het $\langle \tau \rangle_{forward}$')
    axs[2].plot(HomEpi['t'], TauFrontHomAvgs, label = r'Hom $\langle \tau \rangle_{forward}$', ls = '--')
    axs[2].set_ylim(0.5, )
    axs[2].set(title = 'Mean forward generation interval')
    axs[2].legend()
    axs[0].set_xlim(0, 25)
    axs[0].set_ylim(0, )
    axs[1].set_ylim(0, 2.5)
    
    #%% Only homogeneous plot
    beta = 2
    gamma = 1.0
    HomEpi = EpiSimNetMF(beta, gamma, np.array([1]), np.array([1]), 40, 0.01,)
    T = HomEpi['t']
    Taus = np.arange(0, 10, T[1]-T[0])
    TauBackHomAvgs = [mean_BGI_MF(i, Taus, HomEpi['NewPi_I'], beta, gamma) for i in range(0, len(T))]
    TauFrontHomAvgs = [mean_FGI_MF(i, Taus, HomEpi['S'], beta, gamma) for i in range(0, len(T))]
    
    f, axs = plt.subplots(3, 1, sharex = True)
    plt.suptitle(f'Homogeneous model.\n' + 
                 f'$\\gamma$ = {gamma}. ' + 
                 f'$\\beta$ = {beta:.2f}')
    axs[0].plot(HomEpi['t'], HomEpi['I'], label = 'Hom $I$', ls = '--')
    axs[0].plot(HomEpi['t'], HomEpi['S'], label = 'Hom $S$', ls = '--')
    #axs[0].set_yscale('log')
    axs[0].legend()
    
    axs[1].plot(HomEpi['t'], TauBackHomAvgs, label = r'Hom $\langle \tau \rangle_{back}$', ls = '--')
    axs[1].set(title = 'Mean back generation interval')
    axs[1].legend()
    
    axs[2].plot(HomEpi['t'], TauFrontHomAvgs, label = r'Hom $\langle \tau \rangle_{forward}$', ls = '--')
    axs[2].set_ylim(0.5, )
    axs[2].set(title = 'Mean forward generation interval')
    axs[2].legend()
    axs[0].set_xlim(0, 25)
    axs[0].set_ylim(0, )
    axs[1].set_ylim(0, 2.5)
    
    #%% Determine if the contraction coincides with max depletion rate of S in homogeneous model
    
    t_index1 = np.argmin(TauFrontHomAvgs[0:len(TauFrontHomAvgs)//2])
    t_index2 = np.argmax(np.abs(-beta*HomEpi['S']*HomEpi['I']))
    t_index3 = np.argmax(HomEpi['I'])

## Could not conclude anything from this

#%%
# def epidemic_simulator_simple(beta, gamma, rho, max_t, dt):
#     T = np.arange(0, max_t, dt)
#     X_0 = [0, 0, 1]
#     Nk = np.array([1])
#     Sk_0 = Nk*(1-rho)

#     degrees = np.array([1])
#     n = len(Nk)
#     sol = solve_ivp(lambda t, X: dxdt(t, X, rho, beta, gamma, Nk, degrees), t_span=(0, T[-1]), y0 = X_0, method = 'Radau', t_eval = T[1:-1])
#     R, Pi_R, theta, Chi = sol.y[0, :], sol.y[1, :], sol.y[2, :], sol.y[3, :]
#     S = np.exp(-Chi)*np.sum(np.array([Sk_0[i]*theta**degrees[i] for i in range(n)]), 0)        
#     Psioftheta = Psi(theta, Nk, degrees)
#     I = (1-S-R)
#     NewI = beta*S*I
#     #S_approx = Psioftheta*np.exp(-Chi)
#     Pi_S = np.exp(-Chi)*theta*PsiPrime(theta, Nk, degrees)/PsiPrime(1.0, Nk, degrees)
#     Pi_I = 1 - Pi_S -Pi_R
#     #return T, S, I, R, Pi_S, Pi_I, Pi_R, i_rate
#     return sol.t, S, I, R, NewI

