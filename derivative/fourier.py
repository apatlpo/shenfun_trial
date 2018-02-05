"""
Profile different methods in order to compute spatial derivatives
"""
from sys import exit
from sympy import symbols, exp, lambdify
import numpy as np
#import matplotlib.pyplot as plt
from mpi4py import MPI
#
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import Array, Function, TensorProductSpace, project, Dx
from nc_writer import NCWriter
from numba import jit, complex128, float64, int64


try:
    profile  # throws an exception when profile isn't defined
except NameError:
    profile = lambda x: x   # if it'


class method_common():

    def __init__(self, comm, N):
        self.rank = comm.Get_rank()
        self.N = N
        self.T = []

    def init_h(self, hl):
        X = self.T.local_mesh(True)  # physical grid
        self.h = Array(self.T, False)     # in physical space
        self.dh = Array(self.T, False)
        self.h[:] = hl(*X)


class fourier(method_common):

    def __init__(self, comm, N):
        method_common.__init__(self, comm, N)
        # initial spectral spaces
        K0 = C2CBasis(N[0], domain=(-2*np.pi, 2*np.pi))
        K1 = R2CBasis(N[1], domain=(-2*np.pi, 2*np.pi))
        self.T = TensorProductSpace(comm, (K0, K1), **{'planner_effort': 'FFTW_MEASURE'})


class f_project(fourier):

    def __init__(self, comm, N):
        fourier.__init__(self, comm, N)

    def init_h(self, hl):
        fourier.init_h(self, hl)
        self.hf = Function(self.T, False, buffer=self.h)
        #self.dh = np.zeros_like(self.h)

    @profile
    def Dh(self, dvar, dim):
        ''' Wrapper around Dx
        '''
        dvar[:] = self.T.backward(project(Dx(self.hf, dim, 1), self.T))
        return dvar


class f_optimized(fourier):

    def __init__(self, comm, N):
        fourier.__init__(self, comm, N)
        self.work = Array(self.T, True)
        self.K = self.T.local_wavenumbers(scaled=True, eliminate_highest_freq=True)

    @profile
    def Dh(self,dvar,dim):
        ''' Wrapper around Dx
        '''
        self.work = self.T.forward(self.h, self.work)
        #dvar = self.T.backward((1j*self.K[dim])*self.work, dvar)
        dvar = self.T.backward(Kmult(self.K[dim],self.work), dvar)
        return dvar

@profile
def Kmult(K,work):
    return (1j*K)*work

class f_numba(fourier):

    def __init__(self, comm, N):
        fourier.__init__(self, comm, N)
        self.work1 = Array(self.T, True)
        self.work2 = Array(self.T, True)
        self.K = self.T.local_wavenumbers(scaled=True, eliminate_highest_freq=True)

    def init_h(self, hl):
        fourier.init_h(self, hl)
        self.hf = Function(self.T, False, buffer=self.h)

    @profile
    def Dh(self, dvar, dim):
        ''' Wrapper around Dx
        '''
        self.work1 = self.T.forward(self.h, self.work1)
        deriv(self.work1, self.work2, np.squeeze(self.K[0]), np.squeeze(self.K[1]), dim)
        dvar = self.T.backward(self.work2, dvar)
        return dvar

@jit((complex128[:, ::1], complex128[:, ::1], float64[::1], float64[::1], int64), nopython=True)
def deriv(work, wx, kx, ky, dim):
    if dim == 0:
        for i in range(wx.shape[0]):
            k0 = kx[i]*1j
            for j in range(wx.shape[1]):
                wx[i, j] = work[i, j]*k0

    else:
        for i in range(wx.shape[0]):
            for j in range(wx.shape[1]):
                k1 = ky[j]*1j
                wx[i, j] = work[i, j]*k1



if __name__ == '__main__':

    # get MPI info
    comm = MPI.COMM_WORLD


    # Use sympy to set up initial condition
    x, y = symbols("x,y")
    #he = exp(-(x**2 + y**2)/.1**2)
    hl = lambdify((x, y), exp(-(x**2 + y**2)/1.**2), 'numpy')

    # Size of discretization
    N = (128, 128)

    # init methods and
    # loop around methods
    for mclass in [f_project, f_optimized, f_numba]:
        m = mclass(comm,N)
        m.init_h(hl)
        dhdx = np.zeros_like(m.h)
        for i in range(100):
            dhdx = m.Dh(dhdx,0)


    #m = f_project(comm,N)
    #m = f_optimized(comm,N)
    #m = f_numba(comm,N)
    #m.init_h(hl)
    #dhdx = np.zeros_like(m.h)
    #dhdx = m.Dh(dhdx,0)

    #
    #file0 = NCWriter("output.nc", ['dhdx'], m.T, clobber=True)
    #file0.write_tstep(0, dhdx)
