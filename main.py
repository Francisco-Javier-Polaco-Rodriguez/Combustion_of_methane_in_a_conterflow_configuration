import numpy as np
import matplotlib.pyplot as plt
import os

from fluid_solver import *

u_slot = 1
u_coflow = 0.5
N_space = 64
N_time = 21
Lx,Ly = 2e-3,2e-3
dt = 1e-8

[X,Y] = np.meshgrid(np.linspace(0,Lx,N_space),np.linspace(0,Lx,N_space))
up_bc_uy = np.ones(N_space)
down_bc_uy = np.ones(N_space)

## Particula BC of the problem
for k in range(N_space):
    if k < N_space/4:
        up_bc_uy[k] = u_slot
        down_bc_uy[k] = -u_slot
    elif N_space/4 <= k and k < N_space/2:
        up_bc_uy[k] = u_coflow
        down_bc_uy[k] = -u_coflow
    else:
        up_bc_uy[k] = 0
        down_bc_uy[k] = 0

up_bc_ux = np.zeros(N_space)
down_bc_ux = np.zeros(N_space)

## Initial conditions and object creation

u0x = np.zeros([N_space,N_space])
u0y = np.zeros([N_space,N_space])
u0y[0,:]=up_bc_uy
u0y[-1,:]=down_bc_uy
u0x[0,:]=up_bc_ux
u0x[-1,:]=down_bc_ux
p0 = 101225*np.ones([N_space,N_space])


viscosity,density = 15e-6,1.1614
main_fluid = fluid_initial_condition(u_0x = u0x,
                u_0y = u0y,
                p_0 = p0,
                viscosity = viscosity,
                density = density)

plt.show()
bc_ux = boundary_condition(up_bc_ux,down_bc_ux)
bc_uy = boundary_condition(up_bc_uy,down_bc_uy)

solver = pde_fluid_solver(main_fluid,bc_ux,bc_uy,N_time,dt,Lx,Ly)

#solver.presure_solver(np.ones([N_space,N_space]),precision = 0.05,max_reps = 1000)
solver.solve_navier_stokes(N_time,precision_jac = 0.15,repeat_jac = 1000)


# Change this to the path on your oun laptop
path = '/Users/Pacopol/Desktop/Plasma Physics and Fusion Master/Numerical Methods/Project_fluid/Combustion_of_methane_in_a_conterflow_configuration/figures_for_videos'

for k in np.arange(1,N_time,1):
    fig = plt.figure()
    plt.contour(X,Y,solver.p[:,:,k],cmap = 'jet')
    plt.colorbar
    plt.quiver(X,Y,solver.ux[:,:,k],solver.uy[:,:,k])
    fig.savefig(path + '/' + 'frame%i.png'%(k))