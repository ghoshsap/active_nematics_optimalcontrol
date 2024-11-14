
from __future__ import print_function
from copy import deepcopy
from pyclbr import Function
import random

from numpy.lib.shape_base import split
from fenics import *
from dolfin import *
import mshr
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# plt.style.use("/home/chaitanya/git/actnempy/actnempy/prx.mplstyle")
# plt.rc('font', size=16)

set_log_level(50)

def compute_n(Qxx,Qxy):

    S = 2*sqrt( Qxx**2 + Qxy**2)
    Qxx = Qxx/S
    Qxy = Qxy/S
    
    # Evaluate nx and ny from normalized Qxx and Qxy
    nx = sqrt( Qxx + 0.5 )
    ny = Qxy / nx # This ensures ny>0, such that theta lies between 0 to pi
    # nx = nx * sign( Qxy ) # This gives back nx its correct sign.
    return (S, nx, ny)


# MPI stuff
comm = MPI.comm_world
rank = MPI.rank(comm)

count = 0 # Start from `count`
max_iter = 21 # Number of DAL iterations. 

# Test to find appropriate solver
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()


# Define parameters of the simulations
rho = 1.6 # Density
a_2 = (rho-1) 
a_4 = -(1 + rho)/rho**2 # a_2, a_4 from the bulk free energy
lam = 1.0 # Flow alignment parameter
Smax = np.sqrt(2.0) # Maximum of scalar order parameter (Exceeds sometimes?)
mu = 0.01  # Setting friction parameter in stokes equation

seed = 55
######################################

# Optimization parameters (described in [1])
j_val, J2, J3, lmbda, alpha0, u_val = 80, 0.1, 0.1, 0.1, -0.255, 0.5
J4 = 1.1
H = 10

learning_rate, eps = 0.05, 1e-2
########################

# data_dir = f"/Users/saptorshighosh/active_nematics/from_hpcc"
# data_dir = f"/home/fenics/shared"

data_dir = f"/home/fenics/shared"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
nopy = False

# data_dir = f"./data"
# Define geometry and mesh
length = 100
width = 50
channel = mshr.Rectangle(Point(0,0), Point(length,width))
domain = channel
N = int((length+width))
mesh = mshr.generate_mesh(domain, N)


class PeriodicBoundary(SubDomain):

    def inside(self, x, on_boundary):
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], width)) or
                          (near(x[0], length) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], length) and near(x[1], width):
            y[0] = x[0] - length
            y[1] = x[1] - width
        elif near(x[0], length):
            y[0] = x[0] - length
            y[1] = x[1]
        else:   # near(x[1], 127)
            y[0] = x[0]
            y[1] = x[1] - width


bcp = PeriodicBoundary()

tol = 1E-14

# 2D Vector function space to store [Qxx, Qxy] as a vector, since Q is
# traceless and symmetric
FQ = VectorFunctionSpace(mesh, 'P', 1, constrained_domain = bcp)

# Build function space for the Stokes equation. We will use the
# Taylor-Hood elements for this.
# Refer to [2] 

P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH, constrained_domain = bcp)


# 2D Mixed Function Space to store [Qxx, Qxy]:
F = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
MFS = FunctionSpace(mesh, MixedElement([F,F]), constrained_domain = bcp)


FS = FunctionSpace(mesh, F, constrained_domain = bcp)
# Define test function
tf = TestFunction(MFS)
(tx, ty) = split(tf)

#Define Function
psi_new = Function(MFS)
psi_old = Function(MFS)

# Split Mixed Functions
(psixx_new, psixy_new) = split(psi_new)
(psixx_old, psixy_old) = split(psi_old)


# Define test and trial functions
r = TestFunction(MFS)
(rx, ry) = split(r)

# We will call the forward Q vectors as y, and the adjoint of the Q
# vectors as psi

y_new = Function(MFS) # Solution at the next time step
y_old = Function(MFS) # Solution at the current time step
y_star = Function(MFS) # For the target state
(Qxx_new, Qxy_new) = split(y_new)
(Qxx_old, Qxy_old) = split(y_old)
(Qxx_star, Qxy_star) = split(y_star)

Q_new = as_tensor([[Qxx_new, Qxy_new], [Qxy_new, -Qxx_new]])
Q_old = as_tensor([[Qxx_old, Qxy_old], [Qxy_old, -Qxx_old]])
Q_star = as_tensor([[Qxx_star, Qxy_star], [Qxy_star, -Qxx_star]])
R = as_tensor([[rx, ry], [ry, -rx]])


# We will call the forward flow and pressure as u, p and the backward
# ones as nu, phi
(u, p) = TrialFunctions(W)
(nu, phi) = TrialFunctions(W)

(v, q) = TestFunctions(W)
(vv, qq) = TestFunctions(W)

###### Defining an additional scalar function space to store space-dependent
###### activity values and possibly other scalars.

FA = FunctionSpace(mesh, "Lagrange", 1)
# FA = FunctionSpace(mesh, F, 1)
alpha = Function(FA)
alpha_0 = Function(FA)

k = TestFunction(FA)

class InitialConditions_alpha(UserExpression):
    def __init__(self, **kwargs):
        random.seed(1 + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = alpha0

alpha_init = InitialConditions_alpha(degree = 1)
#alpha.interpolate(alpha_init)
alpha_0.interpolate(alpha_init)

####################### Initial and target config for velocity ############################

class InitialConditions_U(UserExpression):
    def __init__(self, **kwargs):
        random.seed(seed + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.0
        values[1] = 0.0
        values[2] = 0.0
    def value_shape(self):
        return (3,)
    
U_init = InitialConditions_U(degree = 1)


class InitialConditions_u(UserExpression):
    def __init__(self, **kwargs):
        random.seed(seed + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = u_val
        values[1] = 0.0
    def value_shape(self):
        return (2,)
    
u_bar = Function(FQ)
u_init = InitialConditions_u(degree = 1)
u_bar.interpolate(u_init)


j_exp = Expression('0.5*j_val*(tanh(x[1] - (mid - (H/2))) - tanh(x[1] - (mid + (H/2))))', element = FS.ufl_element(), j_val = j_val, mid = width/2, H = H)
J1 = interpolate(j_exp, FS)

###############################################################################

# Setup variable and functions to solve the PDEs

# Define symmetric gradient
def epsilon(u):
    return sym(nabla_grad(u))

# Define anti-symmetric gradient
def omega(u):
    return 0.5 * (nabla_grad(u) - nabla_grad(u).T)

# Define active force (with alpha being space-dependent)
Qxxx, Qxxy = grad(alpha*Qxx_new)
Qxyx, Qxyy = grad(alpha*Qxy_new)

divQ = as_vector([Qxxx+Qxyy, Qxyx-Qxxy]) # Divergence of the Q-tensor, computed manually

# Initialize
U = Function(W) # U --> (u0, p)
u0 = split(U)[0]

NU = Function(W) # # NU --> (nu0, phi)
nu0 = split(NU)[0]

############################

# Defining the quantities h_{1-4} as they appear in Eq.(5) of Ref.[1]

h11x = psixx_new*(Qxx_new.dx(0)) + psixy_new*(Qxy_new.dx(0))
h11y = psixx_new*(Qxx_new.dx(1)) + psixy_new*(Qxy_new.dx(1))

h12x = (Qxy_new*psixx_new - Qxx_new*psixy_new).dx(1)
h12y = (Qxx_new*psixy_new - Qxy_new*psixx_new).dx(0)

h13x = lam*(psixx_new.dx(0) + psixy_new.dx(1))
h13y = lam*(psixy_new.dx(0) - psixx_new.dx(1))

ux = +J1*(u0[0] - u_bar[0])
uy = +J1*(u0[1] - u_bar[1])

minus_h1 = as_vector([h11x + h12x + h13x + ux, h11y + h12y + h13y + uy])


h2xx = +(nu0[1].dx(1) - nu0[0].dx(0))
h2xy = -(nu0[0].dx(1) + nu0[1].dx(0))

h4_xx, h4_xy = psixy_new*(u0[0].dx(1) - u0[1].dx(0)), psixx_new*(u0[1].dx(0) - u0[0].dx(1))


############################

# Initial condition for adjoint variable

class InitialConditions_psi(UserExpression):
    def __init__(self, **kwargs):
        random.seed(MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        values[0] = 0.0 
        values[1] = 0.0 
    def value_shape(self):
        return (2,)

psi_init = InitialConditions_psi(degree = 1)

# Initial condition for state variable

NOISE_STRENGTH = 0.01
seed = 35223
class InitialConditionsQ(UserExpression):
    def __init__(self, **kwargs):
        random.seed(seed + MPI.rank(MPI.comm_world))
        super().__init__(**kwargs)
    def eval(self, values, x):
        nx = 1.0 / np.sqrt(2.0)
        ny = 1.0 / np.sqrt(2.0)
        values[0] = (nx**2 - 0.5) + 2 * NOISE_STRENGTH * (0.5 - random.random()) # Qxx
        values[1] = (nx*ny) + 2 * NOISE_STRENGTH * (0.5 - random.random()) # Qxy
        # values[0] = 2 * NOISE_STRENGTH * (0.5 - random.random()) # Qxx
        # values[1] = 2 * NOISE_STRENGTH * (0.5 - random.random()) # Qxy
    def value_shape(self):
        return (2,)

y_init = InitialConditionsQ(degree = 1)

t_total = 800.0
dt = 0.1
num_steps = int(t_total / dt)

#### Creating empty list for storing solutions

y_all = []
adj_all = []
alpha_all = []
u_all = []
nu_all = []
total_cost = []
learning = []

#### Functional forms of cost and derivative of cost with respect to control variable ####

djdalpha = (alpha - alpha_0) - J2*div(grad(alpha)) - (Qxx_new*(nu0[0].dx(0) - nu0[1].dx(1)) + Qxy_new*(nu0[0].dx(1) + nu0[1].dx(0))) + J3*(alpha*lmbda + 1)*exp(lmbda*alpha)
cost_functional = ( 0.5*J1*dot((u0 - u_bar), (u0 - u_bar)) + 0.5*(alpha - alpha_0)**2 + 0.5*J2*dot(grad(alpha), grad(alpha)) + J3*alpha*exp(lmbda*alpha) )*dx

#########################################################################################

Hdf_alpha = HDF5File(comm, f"{data_dir}/restart_3_restart_data_mu_0.01_H_10_alpha_-0.3_t_800.0_J1_80_J4_1.1_J2_0.1_J3_0.1_u_bar_0.5_count_2_length_100_width_50.h5", 'r')

for i in range(8000):

    if(rank == 0 and i % 100 == 0):
        print(f"i:{i}", flush = True)    

    Hdf_alpha.read(alpha, f"alpha/Vector/vector_{i}")
    alpha_all.append(alpha.copy(deepcopy = True))


for i in range(8000, num_steps):

    if(rank ==0 and i % 100 == 0):
        print(f"i:{i}", flush = True)

    alpha_all.append(alpha.copy(deepcopy = True))

Hdf_alpha.close()

def timestep(alpha_all):

    y_all = []
    adj_all = []
    u_all = []
    nu_all = []

    for i in range(num_steps):

        if(rank == 0 and i%1000 == 0):
            print(f"forward_{i}", flush=True)

        alpha.assign(alpha_all[i])

        Res_Q = inner((Q_new - Q_old), R) / dt * dx \
              + inner(dot(u0, nabla_grad(Q_new)), R) * dx \
              + inner((dot(omega(u0), Q_new) - dot(Q_new, omega(u0))), R) * dx \
              + (-lam * inner(epsilon(u0), R) * dx) \
              + (-a_2 * inner(Q_new, R)) * dx \
              + (- 2 * a_4 * (Qxx_old**2 + Qxy_old**2) * inner(Q_new, R)) * dx \
              + 2 * inner(grad(Q_new), grad(R)) * dx 


        solve(Res_Q == 0, y_new)
        y_all.append(y_old.copy(deepcopy = True))
        y_old.assign(y_new)
        
        u_all.append(U.copy(deepcopy = True))

        f = divQ
        a = inner(grad(u), grad(v))*dx + div(v)*p*dx + q*div(u)*dx + mu*inner(u, v)*dx
        L = inner(f, v)*dx

        # Form for use in constructing preconditioner matrix
        b = inner(grad(u), grad(v))*dx + p*q*dx + mu*inner(u, v)*dx

        # Assemble system
        A, bb = assemble_system(a, L, bcs = None)

        # Assemble preconditioner system
        P, btmp = assemble_system(b, L, bcs = None)

        # Create Krylov solver and AMG pre-conditioner
        solver = KrylovSolver(krylov_method, "amg")

        # Associate operator (A) and preconditioner matrix (P)
        solver.set_operators(A, P)
        solver.solve(U.vector(), bb)


    # Initialize adjoint variable as 0.0 at t = tf
    psi_old.interpolate(psi_init)
    (psixx_old, psixy_old) = psi_old.split()
    
    for i in range(num_steps):

        if(rank == 0 and i%1000 == 0):
            print(f"backward:{num_steps - i - 1}", flush=True)
        
        y_new.assign(y_all[num_steps - i - 1])
        U.assign(u_all[num_steps - i - 1])
        alpha.assign(alpha_all[num_steps - i - 1])
    
        Res_psi_x = (psixx_old - psixx_new)*tx/dt*dx + dot(u0, nabla_grad(psixx_new))*tx*dx - 2*dot(grad(psixx_new), grad(tx))*dx \
                  - alpha*h2xx*tx*dx - h4_xx*tx*dx - 4*Qxx_new*a_4*(Qxx_new*psixx_new + Qxy_new*psixy_new)*tx*dx + psixx_new*a_2*tx*dx \
                  - 2*a_4*psixx_new*(Qxx_new**2 + Qxy_new**2)*tx*dx
        Res_psi_y = (psixy_old - psixy_new)*ty/dt*dx + dot(u0, nabla_grad(psixy_new))*ty*dx - 2*dot(grad(psixy_new), grad(ty))*dx \
                  - alpha*h2xy*ty*dx - h4_xy*ty*dx - 4*Qxy_new*a_4*(Qxx_new*psixx_new + Qxy_new*psixy_new)*ty*dx + psixy_new*a_2*ty*dx \
                  - 2*a_4*psixy_new*(Qxx_new**2 + Qxy_new**2)*ty*dx


        Res_psi = (Res_psi_x + Res_psi_y)
        solve(Res_psi == 0, psi_new, bcs = None)
        adj_all.append(psi_old.copy(deepcopy = True))
        psi_old.assign(psi_new)

        nu_all.append(NU.copy(deepcopy = True))

        f = minus_h1

    ##############################################

        a = inner(grad(nu), grad(vv))*dx + div(vv)*phi*dx + qq*div(nu)*dx + mu*inner(nu, vv)*dx
        L = inner(f, vv)*dx

        # Form for use in constructing preconditioner matrix
        b = inner(grad(nu), grad(vv))*dx + phi*qq*dx + mu*inner(nu, vv)*dx

        # Assemble system
        A, bb = assemble_system(a, L, bcs = None)

        # Assemble preconditioner system
        P, btmp = assemble_system(b, L, bcs = None)

        # Create Krylov solver and AMG preconditioner
        solver = KrylovSolver(krylov_method, "amg")

        # Associate operator (A) and preconditioner matrix (P)
        solver.set_operators(A, P)
        solver.solve(NU.vector(), bb)


    adj_all.reverse()
    nu_all.reverse()
    return (y_all, u_all, adj_all, nu_all)


def cost_function(y_all, alpha_all):

    state_cost = 0.0
    # Hdf.read(y_star, f"y_star/Vector/vector_0")

    for i in range(num_steps):
    
        y_new.assign(y_all[i])
        alpha.assign(alpha_all[i])
        U.assign(u_all[i])
        state_cost += dt * assemble(cost_functional)
    
    return (state_cost)


def update_control(y_all, nu_all, alpha_all, lr):

    alphat = Function(FA)
    updated_alpha = []
    alpha_new = Function(FA)

    if(rank == 0):
        print(f"learning-rate:{lr}", flush = True)

    for i in range(num_steps):

        if(rank == 0 and i %1000 == 0):
            print(f"Update:{i}", flush = True)

        NU.assign(nu_all[i])
        y_new.assign(y_all[i])
        alpha.assign(alpha_all[i])
        alphat.assign(project(alpha - lr*djdalpha, FA))

        ####  EXTENSILE CONSTRAINT #### 
        alpha_new = project(0.5*(-sqrt(alphat**2) + alphat), FA)
        alpha_new = project(-sqrt(alpha_new**2), FA)
        ###############################

        updated_alpha.append(alpha_new.copy(deepcopy = True))

    return (updated_alpha)


def armijo(y_all, nu_all, alpha_all):

    arm_cost = 0.0
    Res = -dot(djdalpha, djdalpha)

    for i in range(num_steps):

        y_new.assign(y_all[i])
        NU.assign(nu_all[i])
        alpha.assign(alpha_all[i])
        arm_cost += assemble(Res*dx)

    return (arm_cost)


while(count < max_iter):
    

    data = HDF5File(mesh.mpi_comm(), f'{data_dir}/restart_4_restart_data_mu_{mu}_H_{H}_alpha_{alpha0}_t_{t_total}_J1_{j_val}_J4_{J4}_J2_{J2}_J3_{J3}_u_bar_{u_val}_count_{count}_length_{length}_width_{width}.h5', "w")
    data.write(mesh, "mesh")

    if(rank == 0):
        print(f"count_{count}", flush = True)

    if(count == 0):
        lr = 0.000001*learning_rate
    else:
        lr = learning_rate


    y_new.interpolate(y_init)
    y_old.interpolate(y_init)
    U.interpolate(U_init)

    (y_all, u_all, adj_all, nu_all) = timestep(alpha_all)
    total_cost.append(cost_function(y_all, alpha_all))

    if(count != 0):

        while((total_cost[count] > total_cost[count - 1]) and lr > 1e-5):
            lr = 0.1*lr

            if(rank == 0):
                print(f"Correcting learning rate to:{lr}", flush = True)

            alpha_all = update_control(y_back, nu_back, alpha_back, lr)
            y_new.interpolate(y_init)
            y_old.interpolate(y_init)
            U.interpolate(U_init)

            (y_all, u_all, adj_all, nu_all) = timestep(alpha_all)
            total_cost[count] = cost_function(y_all, alpha_all)

    learning.append(lr)

    armj = armijo(y_all, nu_all, alpha_all)
    alpha_back = alpha_all.copy()
    (y_back, nu_back) = (y_all.copy(), nu_all.copy())
    alpha_all = update_control(y_all, nu_all, alpha_all, lr).copy()

    if(count > 0):
        for i in range(num_steps):    

            U.assign(u_all[i])
            alpha.assign(alpha_back[i])
            y_new.assign(y_all[i])


            data.write(U, f"U/Vector/vector_{i}")
            data.write(alpha, f"alpha/Vector/vector_{i}")
            data.write(y_new, f"y_new/Vector/vector_{i}")
            

    data.close()

    if(count % 1 == 0):
        if(rank == 0):
            np.savez(f"{data_dir}/restart_44_cost_mu_{mu}_alpha_{alpha0}_H_{H}_t_{t_total}_J1_{j_val}_J4_{J4}_J2_{J2}_J3_{J3}_u_bar_{u_val}_count_{count}_length_{length}_width_{width}.npz", cost = total_cost, lr = learning)
    count = count + 1



