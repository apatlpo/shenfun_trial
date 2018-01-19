
from shenfun.chebyshev.bases import ShenBiharmonicBasis 
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.chebyshev.la import Biharmonic as Solver
from shenfun import inner , div , grad , TestFunction , TrialFunction
from shenfun import Function , TensorProductSpace
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

N = (32, 33, 34)
K0 = ShenBiharmonicBasis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
W = TensorProductSpace(comm, (K0, K1, K2))
u = TrialFunction(W)
v = TestFunction(W)
matrices = inner(v, div(grad(div(grad(u))))) 
fj = np.random.random(N)
f_hat = inner(v, fj) # Some right hand side
# or for Legendre:
# matrices = inner(div(grad(v)), div(grad(u))) B = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T) # Solution spectral space 
u_hat = B(u_hat , f_hat) # Solve
u = T.backward(u_hat)

# compute dudx of the solution
#u = T.backward(u_hat) # The original solution on space T 
#K0 = Basis(N[0])
K0 = R2CBasis(N[0]) # this does not work
W0 = TensorProductSpace(comm, (K0, K1, K2)) 
du_hat = project(Dx(u, 0, 1), W0, uh_hat=u_hat)
du = Function(W0)
du = W0.backward(du_hat , du)

