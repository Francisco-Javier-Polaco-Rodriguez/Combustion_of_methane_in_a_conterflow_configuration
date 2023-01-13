import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from scipy.io import savemat

from fluid_solver import *

u_slot = 1
u_coflow = 0.2
N_x,N_y =  32,32
N_time = 200000
T = 20e-6
Lx,Ly = 2e-3,2e-3
dt = T/N_time
viscosity,density = 15e-6,1.1614

[X,Y] = np.meshgrid(np.linspace(0,Lx,N_x),np.linspace(0,Lx,N_y))
up_bc_uy = np.ones(N_x)
down_bc_uy = np.ones(N_x)

## Particula BC of the problem
fact = 2
for k in range(N_x):
    if k < N_x/fact/2:
        up_bc_uy[k] = u_slot
        down_bc_uy[k] = -u_slot
    elif N_x/fact/2 <= k and k < N_x/fact:
        up_bc_uy[k] = u_coflow
        down_bc_uy[k] = -u_coflow
    else:
        up_bc_uy[k] = 0
        down_bc_uy[k] = 0

up_bc_ux = np.zeros(N_x,dtype=np.float32)
down_bc_ux = np.zeros(N_x,dtype=np.float32)

## Initial conditions and object creation

u0x = np.zeros([N_y,N_x],dtype=np.float32)
u0y = np.zeros([N_y,N_x],dtype=np.float32)
u0y[0,:]=up_bc_uy
u0y[-1,:]=down_bc_uy
u0x[0,:]=up_bc_ux
u0x[-1,:]=down_bc_ux
p0 = np.zeros([N_y,N_x],dtype=np.float32)



main_fluid = fluid_initial_condition(u_0x = u0x,
                u_0y = u0y,
                p_0 = p0,
                viscosity = viscosity,
                density = density)

bc_ux = boundary_condition(up_bc_ux,down_bc_ux)
bc_uy = boundary_condition(up_bc_uy,down_bc_uy)

solver = pde_fluid_solver(main_fluid,bc_ux,bc_uy,N_time,dt,Lx,Ly)

# SOLVE EQUATIONS AND SAVE RESULTS
solver.solve_navier_stokes(N_time,precision_jac = 0.05,max_repeat_jac = 1e9,warnig_jacobi = True)
mat = {'ux':solver.ux,'uy':solver.uy,'p':solver.p,'t':solver.dt*np.arange(0,N_time),'X':X,'Y':Y}
path_mat = 'D:/Results of projects/Combustion/Navier Stokes Results'
savemat(path_mat + '/' + 'Simulation_for_%ix%i_grid_and_T=%1.3f_ms.mat'%(N_x,N_y,N_time*solver.dt*1e3),mat)

# Change this to the path on your oun laptop
path = 'D:/Results of projects/Combustion/Navier Stokes Results/Videos'
X,Y = 1e3*X,1e3*Y

Nframes = 200
images = []
frame = 0
N_t_skip_for_vid = np.int32(N_time/Nframes)

for k in tqdm(np.arange(2,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ux = solver.ux[:,:,k]
    uy = solver.uy[:,:,k]
    plt.title('T = %1.1f ns'%(k*dt*1e9))
    # Plot the streamlines with an appropriate colormap and arrow style
    color = np.hypot(ux, uy)
    strp = ax.streamplot(X, Y, ux, uy, color=color, linewidth=1, cmap=plt.cm.inferno,density = 2, arrowstyle='->', arrowsize = 1.5)
    col = plt.colorbar(strp.lines, ax = ax)
    col.set_label('Modulus of the velocity (m/s)')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig,ux,uy

image_folder = path
video_name = path + '/' +'Video_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*solver.dt*1e3)

with imageio.get_writer(video_name + '.mp4',fps = 20) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])

