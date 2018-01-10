r"""
Solve 1 layer shallow water equation [-2pi, 2pi]**3 with periodic bcs

    (u,v)_t = -g*grad(h) - Cd*(u,v)*sqrt(u**2+v**2)        (1)
    h_t = -div( (H+h)*u)                                   (2)

Discretize in time with 4th order Runge-Kutta
with both u(x, y, t=0) and h(x, y, t=0) given.

Using the Fourier basis for all two spatial directions.

mpirun -np 4 python swater_1L.py

"""
from sys import exit
from sympy import symbols, exp, lambdify
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
from time import time
import h5py
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import *
#from spectralDNS.utilities import Timer
from shenfun.utilities.h5py_writer import HDF5Writer
from shenfun.utilities.generate_xdmf import generate_xdmf

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
#timer = Timer()

# parameters
g=1.
H=1.
Cd=.001

# Use sympy to set up initial condition
x, y = symbols("x,y")
he = 1.0*exp(-(x**2 + y**2))
ue = 0.
ve = 0.
ul = lambdify((x, y), ue, 'numpy')
vl = lambdify((x, y), ve, 'numpy')
hl = lambdify((x, y), he, 'numpy')

# Size of discretization
N = (128, 128)

# Defocusing or focusing
gamma = 1

K0 = C2CBasis(N[0], domain=(-2*np.pi, 2*np.pi))
K1 = C2CBasis(N[1], domain=(-2*np.pi, 2*np.pi))
T = TensorProductSpace(comm, (K0, K1), slab=False, **{'planner_effort': 'FFTW_MEASURE'})

TTT = MixedTensorProductSpace([T, T, T])

#Kp0 = C2CBasis(N[0], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
#Kp1 = C2CBasis(N[1], domain=(-2*np.pi, 2*np.pi), padding_factor=1.5)
#Tp = TensorProductSpace(comm, (Kp0, Kp1), slab=False, **{'planner_effort': 'FFTW_MEASURE'})

# Turn on padding by commenting out:
Tp = T

file0 = HDF5Writer("swater_1L{}.h5".format(N[0]), ['u', 'v', 'h'], TTT)

X = T.local_mesh(True)
uvh = Array(TTT, False)
u, v, h = uvh[:]
up = Array(Tp, False)
vp = Array(Tp, False)

duvh = Array(TTT)
du, dv, df = duvh[:]

uvh_hat = Array(TTT)
uvh_hat0 = Array(TTT)
uvh_hat1 = Array(TTT)
w0 = Array(T)
u_hat, v_hat, h_hat = uvh_hat[:]

# initialize
u[:] = ul(*X)
v[:] = vl(*X)
h[:] = hl(*X)
u_hat = T.forward(u, u_hat)
v_hat = T.forward(v, v_hat)
h_hat = T.forward(h, h_hat)

# trial, test functions
uh = TrialFunction(T)
uh_test = TestFunction(T)
vh = TrialFunction(T)
vh_test = TestFunction(T)
hh = TrialFunction(T)
hh_test = TestFunction(T)

# coefficients
A =inner(hh,hh_test)
#A = (2*np.pi)**2
Cu = g*inner(hh, Dx(uh_test, 0, 1))/A
Cv = g*inner(hh, Dx(vh_test, 1, 1))/A
Chu = H*inner(uh, Dx(hh_test, 0, 1))/A
Chv = H*inner(vh, Dx(hh_test, 1, 1))/A

count = 0
def compute_rhs(duvh_hat, uvh_hat, up, vp, T, Tp, w0):
    global count
    count += 1
    duvh_hat.fill(0)
    u_hat, v_hat, h_hat = uvh_hat[:]
    du_hat, dv_hat, dh_hat = duvh_hat[:]
    #
    du_hat[:] = Cu*h_hat
    up = Tp.backward(u_hat, up)
    vp = Tp.backward(v_hat, vp)
    du_hat += -Tp.forward(Cd*up*(up**2+vp**2), w0)  # should be a sqrt here
    #
    dv_hat[:] = Cv*h_hat
    dv_hat += -Tp.forward(Cd*vp*(up**2+vp**2), w0)  # should be a sqrt here
    #
    dh_hat[:] = Chu*u_hat + Chv*v_hat # ignore the nonlinear term for now
    if np.isnan(np.max(np.abs(dh_hat))):
        print('!! blow up')
        exit()
    if np.linalg.norm(dh_hat)==0. or np.linalg.norm(du_hat)==0.:
        print('norm(dhdt)=%.2e or norm(dudt)=%.2e'%(np.linalg.norm(dh_hat), np.linalg.norm(du_hat)))
    return duvh_hat

# Integrate using a 4th order Rung-Kutta method
a = [1./6., 1./3., 1./3., 1./6.]         # Runge-Kutta parameter
b = [0.5, 0.5, 1.]                       # Runge-Kutta parameter
t = 0.0
dt = .001
end_time = 1000.
tstep = 0
write_x_slice = N[0]//2
levels = np.linspace(-1., 1., 100)
#levels = 100
if rank == 0:
    plt.figure()
    image = plt.contourf(X[1][...], X[0][...], h[...], levels)
    plt.draw()
    plt.pause(1e-4)
t0 = time()
#
#K = np.array(T.local_wavenumbers(True, True))
#TV = VectorTensorProductSpace([T, T, T])
#gradu = Array(TV, False)
while t < end_time-1e-8:
    t += dt
    tstep += 1
    print(tstep)
    uvh_hat1[:] = uvh_hat0[:] = uvh_hat
    for rk in range(4):
        duvh = compute_rhs(duvh, uvh_hat, up, vp, T, Tp, w0)
        if rk < 3:
            uvh_hat[:] = uvh_hat0 + b[rk]*dt*duvh
        uvh_hat1 += a[rk]*dt*duvh
    uvh_hat[:] = uvh_hat1

    #timer()

    #if tstep % 10 == 0:
    #    uvh = TTT.backward(uvh_hat, uvh)
    #    file0.write_slice_tstep(tstep, [slice(None), 16], uvh)
    #    file0.write_slice_tstep(tstep, [slice(None), 12], uvh)

    #if tstep % 10 == 0:
    #    uvh = TTT.backward(uvh_hat, uvh)
    #    file0.write_tstep(tstep, uvh)

    if tstep % 100 == 0 and rank == 0:
        uvh = TTT.backward(uvh_hat, uvh)
        image.ax.clear()
        image.ax.contourf(X[1][...], X[0][...], h[...], levels)
        image.ax.set_title('tstep = %d'%(tstep))
        plt.pause(1e-6)
        plt.savefig('swater_1L_{}_real_{}.png'.format(N[0], tstep))

    #if False and tstep % 100 == 0:
    #    uf = TT.backward(uf_hat, uf)
    #    ekin = 0.5*energy_fourier(f_hat, T)
    #    es = 0.5*energy_fourier(1j*K*u_hat, T)
    #    eg = gamma*np.sum(0.5*u**2 - 0.25*u**4)/np.prod(np.array(N))
    #    eg =  comm.allreduce(eg)
    #    gradu = TV.backward(1j*K*u_hat, gradu)
    #    ep = comm.allreduce(np.sum(f*gradu)/np.prod(np.array(N)))
    #    ea = comm.allreduce(np.sum(np.array(X)*(0.5*f**2 + 0.5*gradu**2 - (0.5*u**2 - 0.25*u**4)*f))/np.prod(np.array(N)))
    #    if rank == 0:
    #        print("Time = %2.2f Total energy = %2.8e Linear momentum %2.8e Angular momentum %2.8e" %(t, ekin+es+eg, ep, ea))
    #    comm.barrier()

file0.close()
#timer.final(MPI, rank, True)

if rank == 0:
    generate_xdmf("swater_1L{}.h5".format(N[0]))
