import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import imageio
import os

from fluid_solver import *


path = '/Users/Pacopol/Desktop/Plasma Physics and Fusion Master/Numerical Methods/Project_fluid/Results'
mat = loadmat(path + '/' + 'Simulation_for_32x32_grid_and_T=50.000_ns.mat')
X,Y = mat['X'],mat['Y']
ux,uy = mat['ux'][:,:,-1],mat['uy'][:,:,-1]
p = mat['p']
t = mat['t'][0]
N_x = X.shape[1]
N_y = X.shape[0]


D = 15e-6

N_time = 1000
T = 16e-3
dt = T/N_time


up_bc_N2    = np.ones(N_x)
down_bc_N2  = np.ones(N_x)
up_bc_CH4   = np.ones(N_x)
down_bc_CH4 = np.ones(N_x)
up_bc_O2    = np.ones(N_x)
down_bc_O2  = np.ones(N_x)
up_bc_H2O   = np.ones(N_x)
down_bc_H2O = np.ones(N_x)
up_bc_CO2   = np.ones(N_x)
down_bc_CO2 = np.ones(N_x)


IC_T = np.ones((N_x,N_y))*300
IC_T[np.int32(N_x/4):np.int32(3*N_x/4),:] = 1000


## Particula BC of the problem
fact = 2
for k in range(N_x):

    if k < N_x/fact/2: #slot bc
        up_bc_N2[k]    = 0
        down_bc_N2[k]  = 0.79
        up_bc_CH4[k]   = 1
        down_bc_CH4[k] = 0
        up_bc_O2[k]    = 0
        down_bc_O2[k]  = 0.21
        up_bc_H2O[k]   = 0
        down_bc_H2O[k] = 0
        up_bc_CO2[k]   = 0
        down_bc_CO2[k] = 0

    elif N_x/fact/2 <= k and k < N_x/fact: # Coflow bc
        up_bc_N2[k]    = 1
        down_bc_N2[k]  = 1
        up_bc_CH4[k]   = 0
        down_bc_CH4[k] = 0
        up_bc_O2[k]    = 0
        down_bc_O2[k]  = 0
        up_bc_H2O[k]   = 0
        down_bc_H2O[k] = 0
        up_bc_CO2[k]   = 0
        down_bc_CO2[k] = 0
    else:
        up_bc_N2[k]    = 0
        down_bc_N2[k]  = 0
        up_bc_CH4[k]   = 0
        down_bc_CH4[k] = 0
        up_bc_O2[k]    = 0
        down_bc_O2[k]  = 0
        up_bc_H2O[k]   = 0
        down_bc_H2O[k] = 0
        up_bc_CO2[k]   = 0
        down_bc_CO2[k] = 0

BC = {'N2_up':up_bc_N2,'N2_down':down_bc_N2,'CH4_up':up_bc_CH4,'CH4_down':down_bc_CH4,'O2_up':up_bc_O2,'O2_down':down_bc_O2,'H2O_up':up_bc_H2O,'H2O_down':down_bc_H2O,'CO2_up':up_bc_CO2,'CO2_down':down_bc_CO2}


Lx,Ly = X[-1,-1],Y[-1,-1]
species = difuser_4_species(ux,uy,Lx,Ly,dt,BC,IC_T,D)
species.react(N_time,N_chem = 200)


X,Y = 1e3*X,1e3*Y
path = '/Users/Pacopol/Desktop/Plasma Physics and Fusion Master/Numerical Methods/Project_fluid/Results_chemistry'

Nframes = 50
images = []
frame = 0
N_t_skip_for_vid = np.int32(N_time/Nframes)

###N2
for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    color = ax.pcolormesh(X,Y,species.N2[:,:,k],cmap = 'jet',shading = 'auto')
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
video_name = 'CHEM_N2_difusion_Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*species.dt*1e3)

image_folder = path
video_name = path + '/' + video_name

with imageio.get_writer(video_name + '.mp4',fps = 10) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])

###CH4
frame = 0
images = []
for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    color = ax.pcolormesh(X,Y,species.CH4[:,:,k],cmap = 'jet',shading = 'auto')
    plt.title('T = %1.2f ms'%(k*dt*1e3))
    cbar = plt.colorbar(color)
    cbar.set_label('Molar fraction of $CH_4$')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig

image_folder = path
video_name = 'CHEM_CH4_difusion_Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*species.dt*1e3)

image_folder = path
video_name = path + '/' + video_name

with imageio.get_writer(video_name + '.mp4',fps = 10) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])

###O2
frame = 0
images = []
for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    color = ax.pcolormesh(X,Y,species.O2[:,:,k],cmap = 'jet',shading = 'auto')
    plt.title('T = %1.2f ms'%(k*dt*1e3))
    cbar = plt.colorbar(color)
    cbar.set_label('Molar fraction of $O_2$')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig

image_folder = path
video_name = 'CHEM_O2_difusion_Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*species.dt*1e3)

image_folder = path
video_name = path + '/' + video_name

with imageio.get_writer(video_name + '.mp4',fps = 10) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])

###H2O
frame = 0
images = []
for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    color = ax.pcolormesh(X,Y,species.H2O[:,:,k],cmap = 'jet',shading = 'auto')
    plt.title('T = %1.2f ms'%(k*dt*1e3))
    cbar = plt.colorbar(color)
    cbar.set_label('Molar fraction of $H_2O$')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig

image_folder = path
video_name = 'CHEM_H2O_difusion_Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*species.dt*1e3)

image_folder = path
video_name = path + '/' + video_name

with imageio.get_writer(video_name + '.mp4',fps = 10) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])


###CO2
frame = 0
images = []
for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    color = ax.pcolormesh(X,Y,species.CO2[:,:,k],cmap = 'jet',shading = 'auto')
    plt.title('T = %1.2f ms'%(k*dt*1e3))
    cbar = plt.colorbar(color)
    cbar.set_label('Molar fraction of $CO_2$')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig

image_folder = path
video_name = 'CHEM_CO2_difusion_Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*species.dt*1e3)

image_folder = path
video_name = path + '/' + video_name

with imageio.get_writer(video_name + '.mp4',fps = 10) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])

###T
frame = 0
images = []
for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    color = ax.pcolormesh(X,Y,species.T[:,:,k],cmap = 'jet',shading = 'auto')
    plt.title('T = %1.2f ms'%(k*dt*1e3))
    cbar = plt.colorbar(color)
    cbar.set_label('Temperature (K)')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig

image_folder = path
video_name = 'CHEM_T_difusion_Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms'%(N_x,N_y,N_time*species.dt*1e3)

image_folder = path
video_name = path + '/' + video_name

with imageio.get_writer(video_name + '.mp4',fps = 10) as writer:
    for i in range(len(images)):
        image = imageio.imread(path + '/' + images[i])
        writer.append_data(image)
        os.remove(path + '/' + images[i])
