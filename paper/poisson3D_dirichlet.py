import sys
import importlib
from sympy import symbols, cos, sin, exp, lambdify
import numpy as np
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.tensorproductspace import TensorProductSpace
from shenfun import inner, div, grad, TestFunction, TrialFunction, Function, \
    project, Dx
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD

# Collect basis and solver from either Chebyshev or Legendre submodules
if len(sys.argv) == 3:
    basis = sys.argv[-1]
    N = [eval(sys.argv[-2])]*3
else:
    basis = 'chebyshev'
    N = [eval(sys.argv[-1])]*3
shen = importlib.import_module('.'.join(('shenfun', basis)))
Basis = shen.bases.ShenDirichletBasis
Solver = shen.la.Helmholtz

# Use sympy to compute a rhs, given an analytical solution
x, y, z = symbols("x,y,z")
ue = (cos(4*x) + sin(2*y) + sin(4*z))*(1-y**2)
fe = ue.diff(x, 2) + ue.diff(y, 2) + ue.diff(z, 2)

# Lambdify for faster evaluation
ul = lambdify((x, y, z), ue, 'numpy')
fl = lambdify((x, y, z), fe, 'numpy')

# Size of discretization
#N = [eval(sys.argv[-1])]*3

SD = Basis(N[0])
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
#T = TensorProductSpace(comm, (SD, K1, K2), axes=(0, 1, 2))
T = TensorProductSpace(comm, (K1, K1, K1), axes=(0, 1, 2))
X = T.local_mesh(True)
u = TrialFunction(T)
v = TestFunction(T)

K = T.local_wavenumbers()

if True:
    # inspect data layout
    # see poisson3D example on github
    print(comm.Get_rank(), ' spectral ', T.local_slice()) # spectral space
    print(comm.Get_rank(), ' spectral kx=',K[0][[0,-1],0,0],', ky =',K[1][0,[0,-1],0],' kz =',K[2][0,0,[0,-1]]) # spectral space
    print(comm.Get_rank(), ' physical ', T.local_slice(spectral=False)) # physical space
    print(comm.Get_rank(), ' physical x=',X[0][[0,-1],0,0],', y =',X[1][0,[0,-1],0],' z =',X[2][0,0,[0,-1]]) # physical space
    sys.exit()

# Get f on quad points
fj = fl(*X)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj)
if basis == 'legendre':
    f_hat *= -1.

# Get left hand side of Poisson equation
if basis == 'chebyshev':
    matrices = inner(v, div(grad(u)))
else:
    matrices = inner(grad(v), grad(u))

# Create Helmholtz linear algebra solver
H = Solver(**matrices)

# Solve and transform to real space
u_hat = Function(T)           # Solution spectral space
t0 = time.time()
u_hat = H(u_hat, f_hat)       # Solve
uq = T.backward(u_hat, fast_transform=False)

# Compare with analytical solution
uj = ul(*X)
print("Error=%2.16e" %(np.linalg.norm(uj-uq)))

