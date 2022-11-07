import numpy as np
from colorama import Fore, Back, Style
from tqdm import tqdm

def DDx(M,dx,Bound_left = 0,Bound_right = 0):
    I,J = M.shape
    ddxM = np.zeros([I,J])
    for j in range(1,J-1):
        if j == J-1:
            #ddxM[:,j] = (Bound_right-2*M[:,j]+M[:,j-1])/dx**2
            ddxM[:,j] = ddxM[:,j-1]  ## Intuition. No curvature on the derivative because free wall
        elif j == 0:
            ddxM[:,j] = (M[:,j+1]-2*M[:,j]+Bound_left)/dx**2
        else:
            ddxM[:,j] = (M[:,j+1]-2*M[:,j]+M[:,j-1])/dx**2
    return ddxM

def DDy(M,dy,Bound_up,Bound_down):
    I,J = M.shape
    ddyM = np.zeros([I,J])
    for i in range(1,I-1):
        if i == I-1:
            ddyM[i,:] = (Bound_down-2*M[i,:]+M[i-1,:])/dy**2
        elif i == 0:
            ddyM[i,:] = (M[i+1,:]-2*M[i,:]+Bound_up)/dy**2
        else:
            ddyM[i,:] = (M[i+1,:]-2*M[i,:]+M[i-1,:])/dy**2
    return ddyM

def Dx(M,dx,Bound_left = 0,Bound_right = 0):
    I,J = M.shape
    dxM = np.zeros([I,J])
    for j in range(1,J-1):
        if j == J-1:
            dxM[:,j] = dxM[:,j-1] ## Intuition. No curvature on the derivative because free wall
        elif j == 0:
            dxM[:,j] = (M[:,j+1]-Bound_left)/dx
        else:
            dxM[:,j] = (M[:,j+1]-M[:,j-1])/dx
    return dxM

def Dy(M,dy,Bound_up,Bound_down):
    I,J = M.shape
    dyM = np.zeros([I,J])
    for i in range(1,I-1):
        if i == I-1:
            dyM[i,:] = (Bound_down-M[i-1,:])/dy
        elif i == 0:
            dyM[i,:] = (M[i+1,:]-Bound_up)/dy
        else:
            dyM[i,:] = (M[i+1,:]-M[i-1,:])/dy
    return dyM

## Functions with no bc

def DDx_nobc(M,dx):
    I,J = M.shape
    ddxM = np.zeros([I,J])
    for j in range(1,J-1):
        if j == J-1:
            ddxM[:,j] = ddxM[:,j-1] ## Intuition. No curvature on the derivative because free wall
        elif j == 0:
            ddxM[:,j] = ddxM[:,j+1]
        else:
            ddxM[:,j] = (M[:,j+1]-2*M[:,j]+M[:,j-1])/dx**2
    return ddxM

def DDy_nobc(M,dy):
    I,J = M.shape
    ddyM = np.zeros([I,J])
    for i in range(1,I-1):
        if i == I-1:
            ddyM[i,:] = ddyM[i-1,:]
        elif i == 0:
            ddyM[i,:] = ddyM[i+1,:]
        else:
            ddyM[i,:] = (M[i+1,:]-2*M[i,:]+M[i-1,:])/dy**2
    return ddyM

def Dx_nobc(M,dx):
    I,J = M.shape
    dxM = np.zeros([I,J])
    for j in range(1,J-1):
        if j == J-1:
            dxM[:,j] = dxM[:,j-1] ## Intuition. No curvature on the derivative because free wall
        elif j == 0:
            dxM[:,j] = dxM[:,j+1]
        else:
            dxM[:,j] = (M[:,j+1]-M[:,j-1])/dx
    return dxM

def Dy_nobc(M,dy,):
    I,J = M.shape
    dyM = np.zeros([I,J])
    for i in range(1,I-1):
        if i == I-1:
            dyM[i,:] = dyM[i-1,:]
        elif i == 0:
            dyM[i,:] = dyM[i+1,:]
        else:
            dyM[i,:] = (M[i+1,:]-M[i-1,:])/dy
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

class pde_fluid_solver():
    def __init__(self,fluid_ic,bc_ux,bc_uy,N_time,dt,Lx,Ly,ignore_st = False):
        self.initial_condition = fluid_ic
        self.bc_ux = bc_ux
        self.bc_uy = bc_uy
        self.Nt = N_time
        self.dim = fluid_ic.dimensions
        self.dt = dt
        self.dx = Lx/self.dim[0]
        self.dy = Ly/self.dim[0]
        self.ux= fluid_ic.ux[:,:,np.newaxis]
        self.uy = fluid_ic.uy[:,:,np.newaxis]
        self.p = fluid_ic.pressure[:,:,np.newaxis]
        if self.dim[0]!=bc_ux.dim or self.dim[0]!=bc_uy.dim:
            raise ValueError('Boundary condition and fluid have not the same dimension')
        # Stability comprobation
        Fx = fluid_ic.viscosity*dt/self.dx**2
        Fy = fluid_ic.viscosity*dt/self.dy**2
        Cx = np.mean(np.mean(np.abs(fluid_ic.ux)+np.abs(fluid_ic.uy)))*dt/self.dx
        Cy = Cx*self.dx/self.dy
        if (Fx > 0.25 or Fy > 0.25) and not ignore_st:
            raise ValueError('Invalid fourier number. You need bigger grid or more little time step.  The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy]] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy))
        if Cy > 1 or Cx > 1 and not ignore_st:
            TypeError('Invalid C factor. You need dx/dt of the order of velocities. The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy]] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy))
        print(Fore.BLUE + 'Solver pde class created successfully. The stabilities parameters are  [Fx,Fy] = [%1.3f,%1.3f] [Cx,Cy] =[%1.3f,%1.3f]'%(Fx,Fy,Cx,Cy) + Style.RESET_ALL)
    def presure_solver(self,left_side,right_side,precision = 0.05,max_reps = np.inf):
        p = left_side.copy()
        p_new = p.copy()
        not_good = True
        [I,J] = self.dim
        repeats = 10
        count = 0
        old_rel_error = np.nan
        while not_good and count < max_reps:
            for k in range(repeats):
                    #simpler version commented, but boundary condition in p=p0
                    #for i in range(1,I-1):
                    #    for j in range(1,J-1):
                    #        p_new[i,j] =+0.25*(p[i+1,j]+p_new[i-1,j]+p[i,j+1]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]

                for i in range(I):## Jacobi solver with boundary condition p_0. No boundary condition. We have to take into acount ALL bad i j
                    if i == 0:
                        for j in range(J):
                            if j == 0:
                                p_new[i,j] =+0.25*(p[i+1,j]+p_new[i,j]+p[i,j+1]+p_new[i,j])-0.25*self.dx**2*right_side[i,j]
                            elif j == J-1:
                                p_new[i,j] =+0.25*(p[i+1,j]+p_new[i,j]+p[i,j]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]
                            else:
                                p_new[i,j] =+0.25*(p[i+1,j]+p_new[i,j]+p[i,j+1]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]
                    elif i == I-1:
                        for j in range(J):
                            if j == 0:
                                p_new[i,j] =+0.25*(p[i,j]+p_new[i-1,j]+p[i,j+1]+p_new[i,j])-0.25*self.dx**2*right_side[i,j]
                            elif j == J-1:
                                p_new[i,j] =+0.25*(p[i,j]+p_new[i-1,j]+p[i,j]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]
                            else:
                                p_new[i,j] =+0.25*(p[i,j]+p_new[i-1,j]+p[i,j+1]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]
                    else:
                        for j in range(J):
                            if j == 0:
                                p_new[i,j] =+0.25*(p[i+1,j]+p_new[i-1,j]+p[i,j+1]+p_new[i,j])-0.25*self.dx**2*right_side[i,j]
                            elif j == J-1:
                                p_new[i,j] =+0.25*(p[i+1,j]+p_new[i-1,j]+p[i,j]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]
                            else:
                                p_new[i,j] =+0.25*(p[i+1,j]+p_new[i-1,j]+p[i,j+1]+p_new[i,j-1])-0.25*self.dx**2*right_side[i,j]
                del p
                p = p_new.copy()
                count += 1
            #
            # If first way of calculating, uncomment this error control
            # rel_error = np.mean(np.mean(np.abs(DDx(p_new,self.dx,0,0)+DDy(p_new,self.dy,0,0)-right_side),axis = 0),axis = 0)/np.mean(np.mean(np.abs(right_side),axis = 0),axis = 0)
            #

            rel_error = np.mean(np.mean(np.abs(DDx_nobc(p_new,self.dx)+DDy_nobc(p_new,self.dy)-right_side),axis = 0),axis = 0)/np.mean(np.mean(np.abs(right_side),axis = 0),axis = 0)
            if rel_error > precision and np.abs(rel_error-old_rel_error) < precision*0.1: ## Max accuracy of solver
                not_good = False
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
        if repeats >= max_reps:
            print(Fore.RED + '\nWARNING: the relative error in pressure calculated by Jacobi method is %1.5f bigger than the precision = %1.5f. It is the better convergence that one can reach with  %i repetitions of the method.'%(rel_error,precision,max_reps) + Style.RESET_ALL)
        return p

    def solve_navier_stokes(self,N = np.NaN,precision_jac = 0.1,repeat_jac = 100): # Advance N times Navier Stokes equations.
        if N == np.NaN:
            N = self.N_time
        ## First step. Initialize simulation
        ux = np.zeros([self.dim[0],self.dim[1],N])
        uy = np.zeros([self.dim[0],self.dim[1],N])
        p = np.zeros([self.dim[0],self.dim[1],N])
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
        ## Empty matrix that is gonna be used
        duxuy = np.zeros(self.dim)
        ## Second step simulation !!!
        for k in tqdm(range(1,N)):
            
            ## Step 1 advection
            uxy = ux[:,:,k-1]*uy[:,:,k-1]
            duxuy = Dx(uxy,dx)+Dy(uxy,dy,Bound_up=bc_x_up*bc_y_up,Bound_down=bc_x_down*bc_y_down)
            ux_s  = ux[:,:,k-1]-dt*duxuy
            uy_s = uy[:,:,k-1]-dt*duxuy

            ## Step 2 diffusion
            ux_ss = ux_s+dt*visc*((1+ux[:,:,k-1]**2*dt/visc)*DDx_nobc(ux_s,dx)+(1+uy[:,:,k-1]**2*dt/visc)*DDy_nobc(ux_s,dy))
            uy_ss = uy_s+dt*visc*(DDx_nobc(ux_s,dx)+DDy_nobc(uy_s,dy))

            ##Step 3 compute p
            p[:,:,k]  = self.presure_solver(p[:,:,k-1],dens*dt**-1*(Dx_nobc(ux_ss,dx)+Dy_nobc(uy_ss,dy)),max_reps=repeat_jac,precision=precision_jac)

            ## Step 4 calculate new velocity
            ux[:,:,k] = ux_ss-dt*dens**-1*Dx_nobc(p[:,:,k],dx)
            uy[:,:,k] = uy_ss-dt*dens**-1*Dy_nobc(p[:,:,k],dy)

            # Step 5 Boundary conditions
            ux[0,:,k]  = bc_x_up
            ux[-1,:,k] = bc_x_down
            uy[0,:,k]  = bc_y_up
            uy[-1,:,k] = bc_y_down

            ux[:,0,k]  = np.zeros(self.dim[1])
            # ux[:,-1,k] = nothing needed, free wall
            # uy[:,0,k]  = nothing needed, slipping wall
            # uy[:,-1,k] = nothing needed, free wall
        self.ux = np.concatenate((self.ux,ux),axis = 2)
        self.uy = np.concatenate((self.uy,uy),axis = 2)
        self.p = np.concatenate((self.p,p),axis = 2)

    def solve_navier_stokes_mine(self,N = np.NaN,precision_jac = 0.1,repeat_jac = 100): # Advance N times Navier Stokes equations.
        if N == np.NaN:
            N = self.N_time
        ## First step. Initialize simulation
        ux = np.zeros([self.dim[0],self.dim[1],N])
        uy = np.zeros([self.dim[0],self.dim[1],N])
        p = np.zeros([self.dim[0],self.dim[1],N])
        dt = self.dt
        dx,dy = self.dx,self.dy
        visc = self.initial_condition.viscosity
        dens = self.initial_condition.density
        ux[:,:,0] = self.ux[:,:,-1]
        uy[:,:,0] = self.uy[:,:,-1]
        p[:,:,0]  = self.p[:,:,-1]
        # Preinitialise in RAM memory u star, u star stat
        ux_s = np.zeros(self.dim)
        ux_ss = np.zeros(self.dim)
        uy_s = np.zeros(self.dim)
        uy_ss = np.zeros(self.dim)
        ## Load boundary contition of the problem up and down (velocities in the slot and coflow)
        bc_x_up = self.bc_ux.up
        bc_y_up = self.bc_uy.up
        bc_x_down = self.bc_ux.down
        bc_y_down = self.bc_uy.down
        ## Empty matrix that is gonna be used
        duxuy = np.zeros(self.dim)
        ## Second step simulation !!!
        for k in tqdm(range(1,N)):
            
            ## Step 1 advection
            
            ux_s  = ux[:,:,k-1]-dt*(ux[:,:,k-1]*Dx(ux[:,:,k-1],dx)+uy[:,:,k-1]*Dy(ux[:,:,k-1],dy,Bound_down=bc_x_down,Bound_up=bc_x_up))
            uy_s = uy[:,:,k-1]-dt*(uy[:,:,k-1]*Dx(ux[:,:,k-1],dx)+uy[:,:,k-1]*Dy(uy[:,:,k-1],dy,Bound_down=bc_y_down,Bound_up=bc_y_up))

            ## Step 2 diffusion
            ux_ss = ux_s+dt*visc*(DDx(ux[:,:,k-1],dx)+DDy(ux[:,:,k-1],dy,Bound_down=bc_x_down,Bound_up=bc_x_up))
            uy_ss = uy_s+dt*visc*(DDx(uy[:,:,k-1],dx)+DDy(uy[:,:,k-1],dy,Bound_down=bc_y_down,Bound_up=bc_y_up))

            ##Step 3 compute p
            p[:,:,k]  = -self.presure_solver(p[:,:,k-1],-dens/dt*(Dx_nobc(ux_ss,dx)+Dy_nobc(uy_ss,dy)),max_reps=repeat_jac,precision=precision_jac)

            ## Step 4 calculate new velocity
            ux[:,:,k] = ux_ss-dt*dens**-1*Dx_nobc(p[:,:,k],dx)
            uy[:,:,k] = uy_ss-dt*dens**-1*Dy_nobc(p[:,:,k],dy)
            
            # Step 5 Boundary conditions

            ux[0,:,k]  = bc_x_down
            ux[-1,:,k] = bc_x_up
            uy[0,:,k]  = bc_y_down
            uy[-1,:,k] = bc_y_up

            ux[:,0,k]  = np.zeros(self.dim[1])
            # ux[:,-1,k] = nothing needed, free wall
            # uy[:,0,k]  = nothing needed, slipping wall
            # uy[:,-1,k] = nothing needed, free wall

        self.ux = np.concatenate((self.ux,ux),axis = 2)
        self.uy = np.concatenate((self.uy,uy),axis = 2)
        self.p = np.concatenate((self.p,p),axis = 2)
