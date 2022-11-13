import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.io import loadmat

from fluid_solver import *

mat = loadmat('Simulation_for_124x124_grid_and_T=10.000_ns.mat')
X,Y = mat['X'],mat['Y']
ux,uy = mat['ux'][:,:,-1],mat['uy'][:,:,-1]
p = mat['p']
t = mat['t'][0]
N_x = X.shape[1]
N_y = X.shape[0]

N_slot = 1
N_coflow = 1
D = 15e-6
dt = 1e-6

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

NN = 50000
N2.diffuse_RK4(NN)

# Change this to the path on your oun laptop
path = '/Users/Pacopol/Desktop/Plasma Physics and Fusion Master/Numerical Methods/Project_fluid/Combustion_of_methane_in_a_conterflow_configuration/figures_for_videos'
X,Y = 1e3*X,1e3*Y

Nframes = 100
images = []
frame = 0
N_t_skip_for_vid = np.int32(NN/Nframes)
for k in tqdm(np.arange(1,NN,N_t_skip_for_vid),desc = 'Creating frame'):
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
video_name = 'Video_Nitrogen_diffusion_%ix%i_grid_T=%1.3fms.avi'%(N_x,N_y,NN*N2.dt*1e3)


frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(path + '/' + video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

for image in images:
    os.remove(path + '/' + image)

