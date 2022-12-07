import numpy as np
import matplotlib.pyplot as plt
from ode_chemistry_homogeneous import *




N_chem = 1000
T = 250e-6
dt = T/N_chem
## Order N2,CH4,O2,H2O,CO2

T0 = 1000
N2,CH4,O2,H2O,CO2,T = odesolver_scalar(0.4,0.2,0.4,0,0,T0,dt,N_chem)

t = dt*np.arange(0,N_chem+1)*1e6

fig, ax = plt.subplots(1,2,figsize = (13,26))
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

N2 = np.ones((32,32))
CH4 = np.ones((32,32))
O2 = np.ones((32,32))
H2O = np.ones((32,32))
CO2 =np.ones((32,32))
T = np.ones((32,32))*1e3

print(N2.shape)

N2,CH4,O2,H2O,CO2,T = odesolver(N2,CH4,O2,H2O,CO2,T,dt,N_chem)

print(N2[0,0])
print(CH4[0,0])
print(O2[0,0])
print(H2O[0,0])
print(CO2[0,0])
print(T[0,0])
