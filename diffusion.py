import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from scipy.io import loadmat

from fluid_solver import *
path = '/Users/Pacopol/Desktop/Plasma Physics and Fusion Master/Numerical Methods/Project_fluid/Results'
mat = loadmat(path + '/' + 'Simulation_for_64x64_grid_and_T=50.000_ns.mat')
X,Y = mat['X'],mat['Y']
ux,uy = mat['ux'][:,:,-1],mat['uy'][:,:,-1]
p = mat['p']
t = mat['t'][0]
N_x = X.shape[1]
N_y = X.shape[0]

N_slot = 1
N_coflow = 1
D = 15e-6

N_time = 5000
T = 15e-3
dt = T/N_time
up_bc_N2 = np.ones(N_x)
down_bc_N2 = np.ones(N_x)

## Particula BC of the problem
fact = 2
for k in range(N_x):
    if k < N_x/fact/2:
        up_bc_N2[k] = N_slot
        down_bc_N2[k] = N_slot
    elif N_x/fact/2 <= k and k < N_x/fact:
        up_bc_N2[k] = N_coflow
        down_bc_N2[k] = N_coflow
    else:
        up_bc_N2[k] = 0
        down_bc_N2[k] = 0

bc_N = boundary_condition(up_bc_N2,down_bc_N2)
rho0 = np.zeros([N_x,N_y])
rho0[0,:] = up_bc_N2
rho0[-1,:] = down_bc_N2

Lx,Ly = X[-1,-1],Y[-1,-1]
N2 = diffuser(ux,uy,Lx,Ly,dt,bc_N,rho0,D)




N2.diffuse_RK4_2(N_time)

# Change this to the path on your oun laptop
X,Y = 1e3*X,1e3*Y

Nframes = 100
images = []
frame = 0
N_t_skip_for_vid = np.int32(N_time/Nframes)
for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    color = ax.pcolormesh(X,Y,N2.rho[:,:,k],cmap = 'jet',shading = 'auto')
    plt.title('T = %1.2f ms'%(k*dt*1e3))
    cbar = plt.colorbar(color)
    cbar.set_label('Molar fraction of $N_2$')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig

image_folder = path
video_name = 'RK4_adapted_chemistry_Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*N2.dt*1e3)

image_folder = path
video_name = path + '/' + video_name

with imageio.get_writer(video_name + '.mp4',fps = 20) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])

