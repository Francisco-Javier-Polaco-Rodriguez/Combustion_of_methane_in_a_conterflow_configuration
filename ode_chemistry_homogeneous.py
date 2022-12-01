import numpy as np
import matplotlib.pyplot as plt
from numba import jit

A  = 1.1e8
Ta = 10000
rho = 1.1614
cp = 1200
#@jit(nopython = True)
def Q(CH4,O2,T):
    return A*CH4*O2**2*np.exp(-Ta/T)

#@jit(nopython = True)
def f(CH4,O2,T,W,nu):
    return W*nu*Q(CH4,O2,T)/rho

#@jit(nopython = True)
def RK4(Y_old,CH4,O2,T,W,nu,dt):
    k1 = f(CH4,O2,T,W,nu)
    k2 = f(CH4 + dt*k1/3,O2 + dt*k1/3,T,W,nu)
    k3 = f(CH4 - dt*k1/3 + dt*k2,O2 - dt*k1/3 + dt*k2,T,W,nu)
    k4 = f(CH4 + k1*dt - k2*dt + k3*dt,O2 + k1*dt - k2*dt + k3*dt,T,W,nu)
    return Y_old + dt*k1/8 + 3*dt*k2/8 + 3*dt*k3/8 + dt*k4/8

#@jit(nopython = True)
def fT(T_old,CH4,O2,W,nu):
    return -(h[0]*W[0]**-1*nu[0]+h[1]*W[1]**-1*nu[1]+h[2]*W[2]**-1*nu[2]+h[3]*W[3]**-1*nu[3]+h[4]*W[4]**-1*nu[4])*Q(CH4,O2,T_old)/rho/cp

#@jit(nopython = True)
def RK4_T(T_old,CH4,O2,W,nu,dt):
    k1 = fT(T_old,CH4,O2,W,nu)
    k2 = fT(T_old + dt*k1/3,CH4,O2,W,nu)
    k3 = fT(T_old - dt*k1/3 + dt*k2,CH4,O2,W,nu)
    k4 = fT(T_old + k1*dt - k2*dt + k3*dt,CH4,O2,W,nu)
    return T_old + dt*k1/8 + 3*dt*k2/8 + 3*dt*k3/8 + dt*k4/8

nu = [0,-1,-2,2,1]   ## Stoichiometric coefficients. Order N2,CH4,O2,H2O,CO2
h  = [0,0,-74.9e3,-241.818e3,-393.52e3] ## Entalpy reactions
W  = [28.01340e-3,16.0425e-3,31.99880e-3,18.01528e-3,44.0095e-3]  ## Molar masses (kg/mol)

#@jit(nopython = True)
def odesolver(N2_0,CH4_0,O2_0,H2O_0,CO2_0,T_0,dt,N):
    N2  = np.zeros((N+1))
    CH4 = np.zeros((N+1))
    O2  = np.zeros((N+1))
    H2O = np.zeros((N+1))
    CO2 = np.zeros((N+1))
    T   = np.zeros((N+1))

    N2[0]  = N2_0
    CH4[0] = CH4_0
    O2[0]  = O2_0
    H2O[0] = H2O_0
    CO2[0] = CO2_0
    T[0]   = T_0

    for k in range(1,N+1):
        N2[k]  = RK4(N2[k-1],CH4[k-1],O2[k-1],T[k-1],W[0],nu[0],dt)
        CH4[k] = RK4(CH4[k-1],CH4[k-1],O2[k-1],T[k-1],W[1],nu[1],dt)
        O2[k]  = RK4(O2[k-1],CH4[k-1],O2[k-1],T[k-1],W[2],nu[2],dt)
        H2O[k] = RK4(H2O[k-1],CH4[k-1],O2[k-1],T[k-1],W[3],nu[3],dt)
        CO2[k] = RK4(CO2[k-1],CH4[k-1],O2[k-1],T[k-1],W[4],nu[4],dt)
        T[k]   = RK4_T(T[k-1],CH4[k-1],O2[k-1],W,nu,dt)
    return N2,CH4,O2,H2O,CO2,T


N = 1000
T = 250e-6
dt = T/N
## Order N2,CH4,O2,H2O,CO2

T0 = 1000
N2,CH4,O2,H2O,CO2,T = odesolver(0.4,0.2,0.4,0,0,T0,dt,N)

t = dt*np.arange(0,N+1)*1e6

fig, ax = plt.subplots(1,2)
ax[0].set_xlabel(r'time  ($\mu s$)')
ax[0].plot(t,N2,'-g',label = r'$[N_2]$')
ax[0].plot(t,CH4,'-k',label = r'$[CH_4]$')
ax[0].plot(t,O2,'-b',label = r'$[O_2]$')
ax[0].plot(t,H2O,'-b',label = r'$[H_2O]$',alpha = 0.6)
ax[0].plot(t,CO2,'-y',label = r'$[CO_2]$')
ax[0].legend()
ax[0].set_title('Concentrations')
ax[0].grid()
ax[1].set_xlabel(r'time  ($\mu s$)')
ax[1].plot(t,T,'-r')
ax[1].set_title('Temperature')
ax[1].grid()
plt.show()

