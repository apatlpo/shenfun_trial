
from shenfun.chebyshev.bases import ShenBiharmonicBasis 
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.chebyshev.la import Biharmonic as Solver
from shenfun import inner , div , grad , TestFunction , TrialFunction
from shenfun import Function , TensorProductSpace
from mpi4py import MPI

comm = MPI.COMM_WORLD

N = (32, 33, 34)
K0 = ShenBiharmonicBasis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
W = TensorProductSpace(comm, (K0, K1, K2))
u = TrialFunction(W)
v = TestFunction(W)
matrices = inner(v, div(grad(div(grad(u))))) 
fj=
f_hat = inner(v, fj) # Some right hand side
# or for Legendre:
# matrices = inner(div(grad(v)), div(grad(u))) B = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T) # Solution spectral space 
u_hat = B(u_hat , f_hat) # Solve
u = T.backward(u_hat)


