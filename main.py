import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.io import savemat

from fluid_solver import *

u_slot = 1
u_coflow = 0.5
N_x,N_y = 32,32
N_time = 500
Lx,Ly = 2e-3,2e-3
dt = 2e-9
viscosity,density = 15e-6,1.1614

[X,Y] = np.meshgrid(np.linspace(0,Lx,N_x),np.linspace(0,Lx,N_y))
up_bc_uy = np.ones(N_x)
down_bc_uy = np.ones(N_x)

## Particula BC of the problem
for k in range(N_x):
    if k < N_x/4:
        up_bc_uy[k] = u_slot
        down_bc_uy[k] = -u_slot
    elif N_x/4 <= k and k < N_x/2:
        up_bc_uy[k] = u_coflow
        down_bc_uy[k] = -u_coflow
    else:
        up_bc_uy[k] = 0
        down_bc_uy[k] = 0

up_bc_ux = np.zeros(N_x)
down_bc_ux = np.zeros(N_x)

## Initial conditions and object creation

u0x = np.zeros([N_y,N_x])
u0y = np.zeros([N_y,N_x])
u0y[0,:]=up_bc_uy
u0y[-1,:]=down_bc_uy
u0x[0,:]=up_bc_ux
u0x[-1,:]=down_bc_ux
p0 = np.zeros([N_y,N_x])



main_fluid = fluid_initial_condition(u_0x = u0x,
                u_0y = u0y,
                p_0 = p0,
                viscosity = viscosity,
                density = density)

plt.show()
bc_ux = boundary_condition(up_bc_ux,down_bc_ux)
bc_uy = boundary_condition(up_bc_uy,down_bc_uy)

solver = pde_fluid_solver(main_fluid,bc_ux,bc_uy,N_time,dt,Lx,Ly)

# SOLVE EQUATIONS AND SAVE RESULTS
solver.solve_navier_stokes(N_time,precision_jac = 0.05,repeat_jac = 20000)
mat = {'ux':solver.ux,'uy':solver.uy,'p':solver.p,'t':solver.dt*np.arange(0,N_time),'X':X,'Y':Y}
savemat('Simulation_for_N_space_%i_and_T=%1.3f_ns.mat'%(N_x,N_time*solver.dt),mat)

# Change this to the path on your oun laptop
path = '/Users/Pacopol/Desktop/Plasma Physics and Fusion Master/Numerical Methods/Project_fluid/Combustion_of_methane_in_a_conterflow_configuration/figures_for_videos'
X,Y = 1e3*X,1e3*Y

N_t_skip_for_vid = 2
images = []
frame = 0

for k in tqdm(np.arange(1,N_time,N_t_skip_for_vid),desc = 'Creating frame'):
    fig, ax = plt.subplots(1,1)
    mod_u = np.sqrt(solver.ux[:,:,k]**2+solver.uy[:,:,k]**2)
    color = ax.pcolormesh(X,Y,mod_u,cmap = 'jet',shading = 'auto')
    plt.title('T = %1.2f ns'%(k*dt*1e6))
    plt.quiver(X,Y,solver.ux[:,:,k]/mod_u,solver.uy[:,:,k]/mod_u)
    cbar = plt.colorbar(color)
    cbar.set_label('Modulus of velocity  (m/s)')
    plt.xlabel('x   (mm)',size = 13)
    plt.ylabel('y   (mm)',size = 13)
    plt.axis([0,Lx*1e3-Lx/N_x*1e3,0,Ly*1e3])
    fig.savefig(path + '/' + 'frame%i.png'%(frame))
    images.append('frame%i.png'%(frame))
    frame += 1
    plt.close()
    del fig

image_folder = path
video_name = 'Video_%ix%i_grid_T=%1.3fns.avi'%(N_x,N_y,N_time*solver.dt*1e6)


frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(path + '/' + video_name, 0, 10, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

for image in images:
    os.remove(path + '/' + image)

