from shenfun.chebyshev.bases import ShenDirichletBasis
from shenfun.fourier.bases import FourierBasis
from shenfun import Function , TensorProductSpace
from shenfun import inner , div , grad , TestFunction , TrialFunction
# TensorProductSpace class is used to construct W ,
# Function is a subclass of numpy.ndarray used to hold solution arrays.
from mpi4py import MPI
import numpy as np
from sympy import symbols , sin, cos, lambdify

comm = MPI.COMM_WORLD
N = (32, 33)

K0 = ShenDirichletBasis(N[0])
K1 = FourierBasis(N[1], dtype=np.float)
T = TensorProductSpace(comm, (K0, K1))
print(T)

from shenfun.chebyshev.la import Helmholtz as Solver

# Create a solution that satisfies boundary conditions
x, y = symbols("x,y")
ue = (cos(4*y) + sin(2*x))*(1-x**2)
fe = ue.diff(x, 2) + ue.diff(y, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y), ue, 'numpy') 
fl = lambdify((x, y), fe, 'numpy')
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

# Get f on quad points
fj = fl(X[0], X[1])

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)

# Get left hand side of Poisson equation
matrices = inner(v, div(grad(u)))

# Create Helmholtz linear algebra solver
H = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T) # Solution spectral space 
u_hat = H(u_hat , f_hat) # Solve
u = T.backward(u_hat)

