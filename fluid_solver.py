import numpy as np
from colorama import Fore, Back, Style
from tqdm import tqdm

## DERIVATIVES FOR NAVIER STOKES EQUATIONS. BC ADAPTED TO VELOCITY
from numba import jit


@jit(nopython = True)
def DDx(M:np.array,dx):
    I,J = M.shape
    ddxM = np.zeros((I,J))
    for j in range(1,J-1):
        ddxM[:,j] = (M[:,j+1]-2*M[:,j]+M[:,j-1])/dx**2
    return ddxM

@jit(nopython = True)
def DDy(M:np.array,dy):
    I,J = M.shape
    ddyM = np.zeros((I,J))
    for i in range(1,I-1):
        ddyM[i,:] = (M[i+1,:]-2*M[i,:]+M[i-1,:])/dy**2
    return ddyM

## Note for the future, it is better to make the derivative in 2 steps, first the calculation and finally the boundary conditions, like you did in the second order derivatives
@jit(nopython = True)
def Dx(M:np.array,dx,Bound_left = 0):
    I,J = M.shape
    dxM = np.zeros((I,J))
    for j in range(J):
        if j == J-1:
            #ddxM[:,j] = (Bound_right-2*M[:,j]+M[:,j-1])/dx**2 ## What would be without the bc
            dxM[:,j] = 0 ## Free wall, derivative 0
        elif j == 0:
            dxM[:,j] = (M[:,j+1]-Bound_left)/dx/2
        else:
            dxM[:,j] = (M[:,j+1]-M[:,j-1])/dx/2
    return dxM

@jit(nopython = True)
def Dy(M:np.array,dy,Bound_up,Bound_down):
    I,J = M.shape
    dyM = np.zeros((I,J))
    for i in range(I):
        if i == I-1:
            dyM[i,:] = (Bound_down-M[i-1,:])/dy/2
        elif i == 0:
            dyM[i,:] = (M[i+1,:]-Bound_up)/dy/2
        else:
            dyM[i,:] = (M[i+1,:]-M[i-1,:])/dy/2
    return dyM

## DERIVATIVES WITH GRAD(F)=0 AT BOTH SIDES

@jit(nopython = True)
def Dx_nobc(M:np.array,dx):
    I,J = M.shape
    dxM = np.zeros((I,J))
    for j in range(J):
        if j == J-1:
            dxM[:,j] = 0
        elif j == 0:
            dxM[:,j] = 0
        else:
            dxM[:,j] = (M[:,j+1]-M[:,j-1])/dx/2
    return dxM

@jit(nopython = True)
def Dy_nobc(M:np.array,dy,Bound_down = 0):
    I,J = M.shape
    dyM = np.zeros((I,J))
    for i in range(I):
        if i == I-1:
            dyM[i,:] = (Bound_down-M[i-1,:])/dy/2
        elif i == 0:
            dyM[i,:] = 0
        else:
            dyM[i,:] = (M[i+1,:]-M[i-1,:])/dy/2
    return dyM

## DERIVATIVES ADAPTED TO TRANSPORT OF SPECIES

@jit(nopython = True)
def DDx_transport(M:np.array,dx,Bound_right):
    I,J = M.shape
    ddxM = np.zeros((I,J))
    for j in range(J):
        if j == 0:
            ddxM[:,j] = 0
        elif j == J-1:
            ddxM[:,j] =  0
        else:
            ddxM[:,j] = (M[:,j+1]-2*M[:,j]+M[:,j-1])/dx**2
    return ddxM

@jit(nopython = True)
def DDy_transport(M:np.array,dy,Bound_up,Bound_down):
    I,J = M.shape
    ddyM = np.zeros((I,J))
    J_2 = np.int32(J/2)
    for i in range(I):
        if i == 0:
            ddyM[i,0:J_2] = (M[i+1,0:J_2]-2*M[i,0:J_2]+Bound_up[0:J_2])/dy**2
            ddyM[i,J_2:-1] = 0
        elif i == I-1:
            ddyM[i,0:J_2] = (Bound_down[0:J_2]-2*M[i,0:J_2]+M[i-1,0:J_2])/dy**2
            ddyM[i,J_2:-1] = 0
        else:
            ddyM[i,:] = (M[i+1,:]-2*M[i,:]+M[i-1,:])/dy**2
    return ddyM

@jit(nopython = True)
def Dx_transport(M:np.array,dx,Bound_right):
    I,J = M.shape
    dxM = np.zeros((I,J))
    for j in range(J):
        if j == J-1:
            dxM[:,j] = 0
        elif j == 0:
            dxM[:,j] = 0
        else:
            dxM[:,j] = (M[:,j+1]-M[:,j-1])/dx/2
    return dxM

@jit(nopython = True)
def Dy_transport(M:np.array,dy,Bound_up,Bound_down):
    I,J = M.shape
    dyM = np.zeros((I,J))
    J_2 = np.int32(J/2)
    for i in range(I):
        if i == I-1:
            dyM[i,0:J_2] = (Bound_down[0:J_2]-M[i-1,0:J_2])/dy/2
            dyM[i,J_2:-1] = 0
        elif i == 0:
            dyM[i,0:J_2] = (M[i+1,0:J_2]-Bound_up[0:J_2])/dy/2
            dyM[i,J_2:-1] = 0
        else:
            dyM[i,:] = (M[i+1,:]-M[i-1,:])/dy/2
    return dyM



class fluid_initial_condition():
    def __init__(self,u_0x,u_0y,p_0,viscosity,density):
        self.ux = u_0x
        self.uy = u_0y
        self.pressure = p_0
        self.t = np.array([])
        self.dimensions = u_0x.shape
        self.viscosity = viscosity
        self.density = density
        if p_0.shape != u_0x.shape or u_0x.shape != u_0y.shape:
            raise ValueError('initial pressure and initial velocity must have the same shape')

class boundary_condition():
    def __init__(self,up,down):
        self.up = up
        self.down = down
        self.dim = len(up)


## Precompiling pressure solver
@jit(nopython = True)
def jacobi(p_new:np.array,p:np.array,I:int,J:int,repeats:int,right_side:np.array,dx):
    for k in range(repeats):
        for i in range(1,I-1):
            for j in range(1,J-1):
                p_new[i,j] =0.25*(p[i+1,j]+p_new[i-1,j]+p[i,j+1]+p_new[i,j-1])-0.25*dx**2*right_side[i,j]
        p = p_new.copy()
        p_new[:,0]=p_new[:,1]
        p_new[-1,:]=p_new[-2,:]
        p_new[0,:]=p_new[1,:]
        p_new[:,-1]=0
    return p_new

class pde_fluid_solver():
    def __init__(self,fluid_ic,bc_ux,bc_uy,N_time,dt,Lx,Ly):
        self.initial_condition = fluid_ic
        self.bc_ux = bc_ux
        self.bc_uy = bc_uy
        self.Nt = N_time
        self.dim = fluid_ic.dimensions
        self.dt = dt
        self.X,self.Y = np.meshgrid(np.linspace(0,Lx,self.dim[1]),np.linspace(0,Ly,self.dim[0]))
        self.dx = Lx/self.dim[1]
        self.dy = Ly/self.dim[0]
        self.ux= fluid_ic.ux[:,:,np.newaxis]
        self.uy = fluid_ic.uy[:,:,np.newaxis]
        self.p = fluid_ic.pressure[:,:,np.newaxis]
        if self.dim[1]!=bc_ux.dim or self.dim[1]!=bc_uy.dim:
            raise ValueError('Boundary condition and fluid have not the same dimension')
        if self.dx != self.dy:
            raise ValueError('dx and dy have to be the same.')
        # Stability comprobation
        Fx = fluid_ic.viscosity*dt/self.dx**2
        Fy = fluid_ic.viscosity*dt/self.dy**2
        Cx = np.mean(np.mean(np.abs(fluid_ic.ux)+np.abs(fluid_ic.uy)))*dt/self.dx
        Cy = Cx*self.dx/self.dy
        if Fx > 0.25 or Fy > 0.25:
            raise ValueError('Invalid fourier number. You need bigger grid or more little time step.  The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy]] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy))
        if Cy > 1 or Cx > 1:
            raise TypeError('Invalid C factor. You need dx/dt of the order of velocities. The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy]] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy))
        print(Fore.BLUE + 'Solver pde class created successfully. The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy) + Style.RESET_ALL)
    def presure_solver(self,left_side,right_side,precision = 0.05,max_reps = 10000,warning_pres = True):
        p = left_side.copy()
        p_new = p.copy()
        not_good = True
        [I,J] = self.dim
        repeats = 10
        count = 0
        old_rel_error = np.nan
        while not_good and count < max_reps:
            jacobi(p_new,p,I,J,repeats,right_side,self.dx)
            #for k in range(repeats):
            #    for i in range(1,I-1):
            #        for j in range(1,J-1):
            #            p_new[i,j] =0.25*(p[i+1,j]+p_new[i-1,j]+p[i,j+1]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]
            #    del p
            #    p = p_new.copy()
            #    count += 1
            #    p_new[:,0]=p_new[:,1]
            #    p_new[-1,:]=p_new[-2,:]
            #    p_new[0,:]=p_new[1,:]
            #    p_new[:,-1]=0
            count += repeats
            er_mat = np.abs(DDx(p_new,self.dx)+DDy(p_new,self.dy)-right_side)
            rel_error = np.mean(np.mean(er_mat[1:-1,1:-1],axis = 0),axis = 0)/np.mean(np.mean(np.abs(right_side[1:-1,1:-1]),axis = 0),axis = 0)
            if rel_error > precision and np.abs(rel_error-old_rel_error) < precision*0.1: ## Max accuracy of solver
                not_good = False
                if warning_pres:
                    print(Fore.RED + '\nWARNING: the relative error in pressure calculated by Jacobi method is %1.5f bigger than the precision = %1.5f. It is the better convergence that one can reach with  a %ix%i grid.'%(rel_error,precision,self.dim[0],self.dim[1]) + Style.RESET_ALL)
            if rel_error <=precision: ## Control of the error
                not_good = False
            elif rel_error < 2*precision:
                repeats = int(repeats/1.5)
                if repeats < 50:
                    repeats = 50
            elif rel_error >= 10*precision:
                repeats = int(repeats*2)
            old_rel_error = rel_error
        p = p_new
        if warning_pres:
            print(Fore.RED + '\nWARNING: the relative error in pressure calculated by Jacobi method is %1.5f bigger than the precision = %1.5f. It is the better convergence that one can reach with  %i repetitions of the method.'%(rel_error,precision,max_reps) + Style.RESET_ALL)
        return p
    def solve_navier_stokes(self,N = np.NaN,precision_jac = 0.1,repeat_jac = 100,warnig_jacobi = True): # Advance N times Navier Stokes equations.
        if N == np.NaN:
            N = self.N_time
        ## First step. Initialize simulation
        ux = np.zeros([self.dim[0],self.dim[1],N],dtype=np.float32)
        uy = np.zeros([self.dim[0],self.dim[1],N],dtype=np.float32)
        p = np.zeros([self.dim[0],self.dim[1],N],dtype=np.float32)
        dt = self.dt
        dx,dy = self.dx,self.dy
        visc = self.initial_condition.viscosity
        dens = self.initial_condition.density
        ux[:,:,0] = self.ux[:,:,-1]
        uy[:,:,0] = self.uy[:,:,-1]
        p[:,:,0]  = self.p[:,:,-1]
        # Preinicialize in RAM memory u star, u star stat
        ux_s = np.zeros(self.dim)
        ux_ss = np.zeros(self.dim)
        uy_s = np.zeros(self.dim)
        uy_ss = np.zeros(self.dim)
        ## Load boundary contition of the problem up and down (velocities in the slot and coflow)
        bc_x_up = self.bc_ux.up
        bc_y_up = self.bc_uy.up
        bc_x_down = self.bc_ux.down
        bc_y_down = self.bc_uy.down
        ## Second step simulation !!!
        for k in tqdm(range(1,N),desc = 'Solving Navier-Stokes equations'):
            ## Step 1 ADVECTION
            ux_s = ux[:,:,k-1] - dt*(ux[:,:,k-1]*Dx(ux[:,:,k-1],dx)+uy[:,:,k-1]*Dy(ux[:,:,k-1],dy,Bound_down=bc_x_down*bc_y_down,Bound_up=bc_x_up*bc_y_up))
            uy_s = uy[:,:,k-1] - dt*(ux[:,:,k-1]*Dx(ux[:,:,k-1],dx)+uy[:,:,k-1]*Dy(uy[:,:,k-1],dy,Bound_down=bc_y_down**2,Bound_up=bc_y_up**2))

            ## ARTIFICIAL DIFFUSION, order 2 in advection
            ux_s = ux_s + 0.5*dt**2*(ux[:,:,k-1]**2*DDx(ux[:,:,k-1],dx)+uy[:,:,k-1]**2*DDy(ux[:,:,k-1],dy))
            ux_s = ux_s + 0.5*dt**2*(ux[:,:,k-1]**2*DDx(uy[:,:,k-1],dx)+uy[:,:,k-1]**2*DDy(uy[:,:,k-1],dy))

            ## Step 2 DIFFUSION
            ux_ss = ux_s + dt*visc*(DDx(ux_s,dx)+DDy(ux_s,dy))
            uy_ss = uy_s + dt*visc*(DDx(ux_s,dx)+DDy(uy_s,dy))

            ## Step 3 IMPOSE COMPRESSIBILITY 
            p[:,:,k]  = self.presure_solver(p[:,:,k-1],dens*(Dx(ux_ss,dx)+Dy(uy_ss,dy,Bound_down=bc_y_down,Bound_up=bc_y_up))/dt,max_reps=repeat_jac,precision=precision_jac,warning_pres = warnig_jacobi)

            ##Step 4, ADVANCE TIME
            ux[:,:,k] = ux_ss-dt*Dx_nobc(p[:,:,k],dx)/dens
            uy[:,:,k] = uy_ss-dt*Dy_nobc(p[:,:,k],dy)/dens

            # Extra step. No boundary condition in derivatives. We impose here the boundary conditions
            ux[0,:,k]  = bc_x_up
            ux[-1,:,k] = bc_x_down
            uy[0,:,k]  = bc_y_up
            uy[-1,:,k] = bc_y_down

            ux[:,0,k]  = np.zeros(self.dim[0])
            ux[:,-1,k] = ux[:,-2,k] # The velocity from left is advected, otherwise we have always ux()
            uy[:,-1,k] = uy[:,-2,k]
            # uy[:,0,k]  = nothing needed, slipping wall
            # uy[:,-1,k] = nothing needed, free wall
        self.ux = np.concatenate((self.ux,ux),axis = 2)
        self.uy = np.concatenate((self.uy,uy),axis = 2)
        self.p = np.concatenate((self.p,p),axis = 2)

class diffuser():
    def __init__(self,ux,uy,Lx,Ly,dt_steady_sim,bc,rho0,diff_coef):
        self.bc_up = bc.up
        self.bc_down = bc.down
        self.rho = rho0[:,:,np.newaxis]
        self.ux = ux
        self.uy = uy
        self.Lx = Lx
        self.Ly = Ly
        self.dim = ux.shape
        self.diff_coef = diff_coef
        self.dy = Ly/self.dim[0]
        self.dx = Lx/self.dim[1]
        self.dt = dt_steady_sim
        # Stability comprobation
        Fx = diff_coef*self.dt/self.dx**2
        Fy = diff_coef*self.dt/self.dy**2
        Cx = np.mean(np.mean(np.mean(np.abs(ux)+np.abs(uy))))*self.dt/self.dx
        Cy = Cx*self.dx/self.dy
        if Fx > 0.25 or Fy > 0.25:
            raise ValueError('Invalid fourier number. You need bigger grid or more little time step.  The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy]] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy))
        if Cy > 1 or Cx > 1:
            TypeError('Invalid C factor. You need dx/dt of the order of velocities. The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy]] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy))
        print(Fore.BLUE + 'Solver pde class created successfully. The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy) + Style.RESET_ALL)

    def function_scheme(self,phi,ux,uy,D,dx,dy,dt):
        dxphi  = Dx_transport(phi,dx,Bound_right = 0)
        dx2phi = DDx_transport(phi,dx,Bound_right = 0)
        dyphi  = Dy_transport(phi,dy,Bound_up = self.bc_up,Bound_down = self.bc_down)
        dy2phi = DDy_transport(phi,dy,Bound_up = self.bc_up,Bound_down = self.bc_down)
        return D*(dx2phi+dy2phi) + ux**2*dt*dx2phi/2 + uy**2*dt*dy2phi/2 - ux*dxphi - uy*dyphi

    def diffuse_RK4(self,N):
        rho = np.zeros([self.dim[1],self.dim[0],N])
        rho[:,:,0] = self.rho[:,:,-1]
        dx = self.dx
        dy = self.dy
        dt = self.dt
        D = self.diff_coef
        for k in tqdm(range(1,N)):
            k1 = self.function_scheme(rho[:,:,k-1],self.ux,self.uy,D,dx,dy,dt)
            k2 = self.function_scheme(rho[:,:,k-1]+dt*k1/3,self.ux,self.uy,D,dx,dy,dt)
            k3 = self.function_scheme(rho[:,:,k-1]-dt*k1/3+dt*k2,self.ux,self.uy,D,dx,dy,dt)
            k4 = self.function_scheme(rho[:,:,k-1]+dt*k1-dt*k2+dt*k3,self.ux,self.uy,D,dx,dy,dt)
            rho[:,:,k] = rho[:,:,k-1] + dt*k1/8 + 3*dt*k2/8 + 3*dt*k3/8 + dt*k4/8
            rho[:,-1,k] = rho[:,-2,k]
        self.rho = np.concatenate((self.rho,rho[:,:,1:-1]),axis = 2)