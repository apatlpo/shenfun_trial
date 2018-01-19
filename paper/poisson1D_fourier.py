from sympy import Symbol , cos
import numpy as np
from shenfun import inner , div , grad , TestFunction , TrialFunction 
from shenfun.fourier.bases import FourierBasis

# Use Sympy to compute a rhs, given an analytical solution
x = Symbol("x")
ue = cos(4*x)
fe = ue.diff(x, 2)

# Create Fourier basis with N basis functions
N = 32
ST = FourierBasis(N, np.float, plan=True)
u = TrialFunction(ST)
v = TestFunction(ST)
X = ST.mesh(N)

# Get f and exact solution on quad points
fj = np.array([fe.subs(x, j) for j in X], dtype=np.float) 
uj = np.array([ue.subs(x, i) for i in X], dtype=np.float)

# Assemble right and left hand sides
f_hat = inner(v, fj)  # array 
A = inner(v, div(grad(u))) # matrix

# Solve Poisson equation
u_hat = A.solve(f_hat)

# Transfer solution back to real space
uq = ST.backward(u_hat)
assert np.allclose(uj, uq)

### pure Fourier approach

# Transform right hand side
f_hat = ST.forward(fj)
# Wavenumers
k = ST.wavenumbers(N)
k[0] = 1
# Solve Poisson equation (solution in f_hat)
f_hat /= k**2
uq2 = ST.backward(f_hat)

assert np.allclose(uq, uq2)



