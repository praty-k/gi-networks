# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:18:36 2024

@author: pkollepara
"""

# Package importation
import numpy as np 
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

#%% exact solution object definition
class QuenchedSolution(object):
    def __init__(self, beta, gamma, degrees, Nk, max_t, dt, t, S, I, R, Phi_S, Phi_I, Phi_R, NewPhi_I, NewI, theta):
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
        self.Phi_S = Phi_S
        self.Phi_I = Phi_I
        self.Phi_R = Phi_R
        self.NewPhi_I = NewPhi_I
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

def dxdt(t, X, beta, gamma, Nk, degrees):
    theta, R = X
    S = Psi(theta, Nk, degrees)
    phi_S = PsiPrime(theta, Nk, degrees)/PsiPrime(1.0, Nk, degrees)
    I = 1 - S - R
    return np.array([-beta*theta + beta*phi_S + gamma*(1-theta), gamma*I])

def FS(x, G, n): 
    return [x[k] - n[k]*(1-np.exp(-np.sum(G[k]*x)/n[k])) for k in range(len(n))]

# Fxn for mean backward generatio interval
def mean_BGI_Q(t_index, Taus, rate_new_infections, beta, gamma, draw = False):
    N = len(Taus)
    dt = Taus[1] - Taus[0]
    g_Taus = (beta+gamma)*np.exp(-(beta+gamma)*Taus)
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
    TauSquareAvg = np.sum(b_Taus*Taus**2*dt)
    # TauSDev = np.sqrt(TauSquareAvg-TauAvg**2)
    # b_TausCDF = np.cumsum(b_Taus*dt)
    # TauUL = Taus[b_TausCDF>=0.80][0]
    # TauLL = Taus[b_TausCDF>=0.20][0]
    if draw:
        plt.plot(Taus, b_Taus, label = 'BGI distribution')
        plt.plot(Taus, g_Taus, label = 'IGI distribution')
        plt.title(f'{t_index = }')
        plt.legend()
        return (TauAvg, Rates, RatesSum, b_Taus)
    else:
        return TauAvg#, TauUL, TauLL

#Fxn for mean forward generation intervals
def mean_FGI_Q(t_index, Taus, Pi_S, beta, gamma, draw = False):
    N = len(Taus)
    dt = Taus[1] - Taus[0]
    g_Taus = (beta+gamma)*np.exp(-(beta+gamma)*Taus)
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
    TauSquareAvg = np.sum(f_Taus*Taus**2*dt)
    #TauSDev = np.sqrt(TauSquareAvg-TauAvg**2)
    #f_TausCDF = np.cumsum(f_Taus*dt)
    #TauUL = Taus[f_TausCDF>=0.80][0]
    #TauLL = Taus[f_TausCDF>=0.20][0]
    
    if draw:
        plt.plot(Taus, f_Taus, label = 'FGI distribution')
        plt.plot(Taus, g_Taus, label = 'IGI distribution')
        plt.title(f'{t_index = }')
        plt.legend()
        return (TauAvg, Rates, RatesSum, f_Taus)
    else:
        return TauAvg#, TauUL, TauLL

#Fxn for forward reproduction number
def FRN_Q(t_index, Taus, Phi_S, theta, Nk, degrees, beta, gamma):
    N = len(Taus)
    dt = Taus[1] - Taus[0]
    p_Taus = np.exp(-(beta+gamma)*Taus)
    """denominator = np.sum([dt*p_Taus[i]*rate_new_infections[t_index - i] for i in range(0, N)])"""
    Rates = []
    for i in range(0, N):
        if t_index + i < len(Phi_S):
            Rates.append(p_Taus[i]*Phi_S[t_index + i])
        else:
            Rates.append(0)
    
    RatesSum = np.sum(Rates)*dt
    f_Taus = Rates/RatesSum
    R_f = beta*PsiDoublePrime(theta[t_index], Nk, degrees)/PsiPrime(theta[t_index], Nk, degrees)*RatesSum
    
    return R_f


##
def EpiSimNetQ(beta, gamma, degrees, Nk, max_t, dt, theta_0 = 1-1e-4, method = 'Radau'):
    T = np.arange(0, max_t, dt)
    #theta_0 = 1-1e-4
    X_0 = [theta_0, 0]
    sol = solve_ivp(lambda t, X: dxdt(t, X, beta, gamma, Nk, degrees), t_span=(0, T[-1]), y0 = X_0, method = 'Radau', t_eval = T[1:-1])
    theta, R = sol.y[0, :], sol.y[1, :]
    S = Psi(theta, Nk, degrees)
    I = 1 - S - R
    Phi_S = PsiPrime(theta, Nk, degrees)/PsiPrime(1, Nk, degrees)
    Phi_R = gamma/beta*(1-theta)
    Phi_I = theta - Phi_S - Phi_R
    NewPhi_I = beta*Phi_I*PsiDoublePrime(theta, Nk, degrees)/PsiPrime(1, Nk, degrees)
    NewI = beta*Phi_I*PsiPrime(theta, Nk, degrees)
    #return T, S, I, R, Pi_S, Pi_I, pi_R, i_rate
    
    epidemic_dict = {'t': sol.t, # Just a back up in case the object does not work as expected
                'S': S, 
                'I': I, 
                'R': R, 
                'Phi_S': Phi_S, 
                'Phi_I': Phi_I, 
                'Phi_R': Phi_R, 
                'NewPhi_I': NewPhi_I, 
                'theta': theta
                }
    epidemic = QuenchedSolution(beta, gamma, degrees, Nk, max_t, dt, sol.t, S, I, R, Phi_S, Phi_I, Phi_R, NewPhi_I, NewI, theta)
    return epidemic

#%%
if __name__ == '__main__':
    #%%% Heterogeneous SIR on 
    
    #Power law
    alpha = 2 #Exponent
    cutoff = 30 #Maximum degree in the network that is allowed
    degrees = np.array(list(range(1, cutoff+1))).astype(float)
    Nk = degrees**-alpha
    Nk = Nk/np.sum(Nk)
    
    beta = .45 # R0 approx 2
    gamma = 1 
    
    DegreeSquareAvg = np.sum(Nk*degrees**2)
    DegreeAvg = np.sum(Nk*degrees)
    R0 = beta/(beta+gamma)*(DegreeSquareAvg-DegreeAvg)/DegreeAvg
    HetEpi = EpiSimNetQ(beta, gamma, degrees, Nk, 25, 0.01,)
    T, S, I, R, Phi_S, Phi_I, Phi_R, NewPhi_I, theta = [HetEpi[key] for key in HetEpi.keys()]
    
#%%
    Taus = np.arange(0, 10, T[1]-T[0])
    TauBackAvgs = [mean_BGI_Q(i, Taus, NewPhi_I, beta, gamma) for i in range(0, len(T))]
    TauFrontAvgs = [mean_FGI_Q(i, Taus, Phi_S, beta, gamma) for i in range(0, len(T))] # Forward generation average
    
#%%
    f, axs = plt.subplots(3, 1, sharex = True, figsize = (4, 6))
    plt.suptitle(f'Power-law like network. k_max = {cutoff}. $\\beta$ = {beta:.2f}. $\\gamma$ = {gamma}')
    
    axs[0].plot(T, NewPhi_I, label = '$\phi_I$')
    axs[0].plot(T, S, label = '$S$')
    axs[0].plot(T, Phi_S, label = '$\phi_S$')
    #axs[0].set_yscale('log')
    axs[0].legend()
    axs[1].plot(T, TauBackAvgs, label = r'$\langle \tau \rangle_{back}$')
    axs[1].set(title = 'Mean back generation interval')
    axs[1].legend()
    axs[2].plot(T, TauFrontAvgs, label = r'$\langle \tau \rangle_{front}$')
    #axs[2].set_ylim(0.5, )
    axs[2].set(title = 'Mean forward generation interval')
    axs[2].legend()
    axs[0].set_xlim(0, 25)
    #axs[0].set_ylim(0, )
    #axs[1].set_ylim(0, 2)
    
#%%
    
    #Homogeneous
    k = 8
    degrees = np.array([k])
    Nk = np.array([1])
    
    beta_eff = R0/(k-1)
    
    
    HomEpi = EpiSimNetQ(beta_eff, gamma, degrees, Nk, 25, 0.01,)
    T, S, I, R, Phi_S, Phi_I, Phi_R, NewPhi_I, theta = [HomEpi[key] for key in HetEpi.keys()]
    
#%%
    Taus = np.arange(0, 10, T[1]-T[0])
    TauBackAvgs = [mean_BGI_Q(i, Taus, NewPhi_I, beta_eff, gamma) for i in range(0, len(T))]
    TauFrontAvgs = [mean_FGI_Q(i, Taus, Phi_S, beta_eff, gamma) for i in range(0, len(T))] # Forward generation average
    
#%%
    f, axs = plt.subplots(3, 1, sharex = True, figsize = (4, 6))
    plt.suptitle(f'Homogeneous network. k = {k}. $\\beta_{{eff}}$ = {beta_eff:.2f}. $\\gamma$ = {gamma}')
    
    axs[0].plot(T, NewPhi_I, label = '$\Phi_I$')
    axs[0].plot(T, S, label = '$S$')
    axs[0].plot(T, Phi_S, label = '$\Phi_S$')
    #axs[0].set_yscale('log')
    axs[0].legend()
    axs[1].plot(T, TauBackAvgs, label = r'$\langle \tau \rangle_{back}$')
    axs[1].set(title = 'Mean back generation interval')
    axs[1].legend()
    axs[2].plot(T, TauFrontAvgs, label = r'$\langle \tau \rangle_{front}$')
    #axs[2].set_ylim(0.5, )
    axs[2].set(title = 'Mean forward generation interval')
    axs[2].legend()
    axs[0].set_xlim(0, 25)
    #axs[0].set_ylim(0, )
    #axs[1].set_ylim(0, 2)