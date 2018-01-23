from shenfun.chebyshev.bases import ShenBiharmonicBasis
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun import Function , TensorProductSpace, VectorTensorProductSpace , curl
from shenfun import inner , curl , TestFunction
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

N = (32, 33, 34)
K0 = ShenBiharmonicBasis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])

T = TensorProductSpace(comm, (K0, K1, K2))
Tk = VectorTensorProductSpace([T, T, T])

v = TestFunction(Tk)
u_ = Function(Tk, False)
u_[:] = np.random.random(u_.shape)
u_hat = Function(Tk)
u_hat = Tk.forward(u_, u_hat)
w_hat = inner(v, curl(u_), uh_hat=u_hat)
