"""
Solve nonlinear 1 layer shallow water equation [-L/2., L/2.]**2 with periodic bcs

    (u,v)_t + (u.nabla)(u,v) - f (u,-v) = -g*grad(h) - Cd/H*(u,v)*sqrt(u**2+v**2)        (1)
    h_t = -div( (H+h)*u)                                                    (2)

Boundary conditions are of the following types:
u(x=+-L/2)=0 (no flow through bdy)
dudy(y=+-L/2)=0 (should be parametrized friction)
dvdx(x=+-L/2)=0 (should be parametrized friction)
v(y=+-L/2)=0 (no flow through bdy)
Neumann for h (no rationale)

Equations are timestepped in physical space though.

Discretize in time with 4th order Runge-Kutta
with both u(x, y, t=0) and h(x, y, t=0) given.

mpirun -np 4 python swater_1L_physical_closed.py

For profiling:
kernprof -lv swater_1L_physical_closed.py
python -m line_profiler swater_1L_physical_closed.py.lprof


"""
from sys import exit
from sympy import symbols, exp, lambdify
import matplotlib.pyplot as plt
from mpi4py import MPI
#
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import *

# get MPI info
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# physical parameters
g=1.
L=1000*1.e3
H=4000.
f=7.e-5
Cd=.001 # inverse length scale

# Use sympy to set up initial condition
x, y = symbols("x,y")
Le=500.*1e3
he = 1.0*exp(-(x**2 + y**2)/Le**2)
ue = 0.
ve = 0.
ul = lambdify((x, y), ue, 'numpy')
vl = lambdify((x, y), ve, 'numpy')
hl = lambdify((x, y), he, 'numpy')

# Size of discretization
N = (128, 128)

# initial spectral spaces
SD0 = chebyshev.bases.ShenDirichletBasis(N[0], domain=(-2*np.pi*L, 2*np.pi*L))
SD1 = chebyshev.bases.ShenDirichletBasis(N[1], domain=(-2*np.pi*L, 2*np.pi*L))
SN0 = chebyshev.bases.ShenNeumannBasis(N[0], domain=(-2*np.pi*L, 2*np.pi*L))
SN1 = chebyshev.bases.ShenNeumannBasis(N[1], domain=(-2*np.pi*L, 2*np.pi*L))
#
Tu = TensorProductSpace(comm, (SD0, SN1), **{'planner_effort': 'FFTW_MEASURE'})
Tv = TensorProductSpace(comm, (SN0, SD1), **{'planner_effort': 'FFTW_MEASURE'})
Th = TensorProductSpace(comm, (SN0, SN1), **{'planner_effort': 'FFTW_MEASURE'})
#
TTT = MixedTensorProductSpace([Tu, Tv, Th])


# init vectors
X = Th.local_mesh(True)  # physical grid
uvh = Array(TTT, False) # in physical space
u, v, h = uvh[:]
print(u.shape)
print(v.shape)
print(h.shape)
up = Array(Tu, False)
vp = Array(Tv, False)

# for time stepping, everything is in physical space
uvh0 = Array(TTT, False)
uvh1 = Array(TTT, False)
duvh = Array(TTT, False)
du, dv, df = duvh[:]

# initialize
u[:] = ul(*X)
v[:] = vl(*X)
h[:] = hl(*X)

print(u.shape)
print(v.shape)
print(h.shape)

# init variables for derivatives
uf = Function(Tu, False, buffer=u)
vf = Function(Tv, False, buffer=v)
hf = Function(Th, False, buffer=h)
dv0 = np.zeros_like(u)
def D(varf,dvar,dim,T):
    ''' Wrapper around Dx
    T is final space
    '''
    dvar[:] = T.backward(project(Dx(varf, dim, 1), T), output_array=dvar)
    return dvar

dhdx = D(hf,dv0,0,Th)
#print(dhdx is dv0) # test if objects are the same

if rank == 0:
    plt.figure()
    levels = np.linspace(-1., 1., 100)
    image = plt.contourf(X[1][...], X[0][...], dhdx[...], 10)
    plt.draw()
    plt.pause(1e2)

# RHS
count = 0
#@profile
def compute_rhs(duvh, uvh):
    global count
    count += 1
    duvh.fill(0)
    u, v, h = uvh[:]
    du, dv, dh = duvh[:]
    #
    du[:] = -g*D(hf,dv0,0,Tu) # -g*dhdx
    du += -u*D(uf,dv0,0,Tu)   # -u*dudx
    du += -v*D(uf,dv0,1,Tu)   # -v*dudy
    du += -Cd/H*u*np.sqrt(u**2+v**2)
    du += f*v
    #
    dv[:] = -g*D(hf,dv0,1,Tv) # -g*dhdy
    dv += -u*D(vf,dv0,0,Tv)   # -u*dvdx
    dv += -v*D(vf,dv0,1,Tv)   # -v*dvdy
    dv += -Cd/H*v*np.sqrt(u**2+v**2)
    dv += -f*u
    #
    dh[:] = -(H+h)*D(uf,dv0,0,Th) # -H*dudx
    dh += -u*D(hf,dv0,0,Th)   # -u*dhdx
    dh += -v*D(hf,dv0,1,Th)   # -v*dhdy
    dh += -H*D(vf,dv0,1,Th)   # -H*dvdy
    if np.isnan(np.max(np.abs(dh))):
        print('!! blow up')
        exit()
    if np.linalg.norm(dh)==0. or np.linalg.norm(du)==0.:
        print('norm(dhdt)=%.2e or norm(dudt)=%.2e'%(np.linalg.norm(dh), np.linalg.norm(du)))
    return duvh


# Integrate using a 4th order Rung-Kutta method
a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter
#
t = 0.0
dt = 60.*10
end_time = 3600.*24
tstep = 0
#
if rank == 0:
    plt.figure()
    image = plt.contourf(X[1][...], X[0][...], h[...], levels)
    plt.draw()
    plt.pause(1.e-4)
#
while t < end_time-1e-8:
    t += dt
    tstep += 1
    print(tstep)
    uvh1[:] = uvh0[:] = uvh
    for rk in range(4):
        duvh = compute_rhs(duvh, uvh)
        if rk < 3:
            uvh[:] = uvh + b[rk]*dt*duvh
        uvh1 += a[rk]*dt*duvh
    uvh[:] = uvh1

    if tstep % 10 == 0 and rank == 0:
        image.ax.clear()
        image.ax.contourf(X[1][...], X[0][...], h[...], levels)
        image.ax.set_title('tstep = %d'%(tstep))
        plt.pause(1e-6)
        plt.savefig('figs/swater_1L_{}_real_{}.png'.format(N[0], tstep))
