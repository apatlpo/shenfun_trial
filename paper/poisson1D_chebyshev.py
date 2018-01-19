from sympy import Symbol , sin, lambdify
import numpy as np
from shenfun import inner , div , grad , TestFunction , TrialFunctio
from shenfun.chebyshev.bases import ShenDirichletBasis

# Use sympy to compute a rhs, given an analytical solution
x = Symbol("x")
ue = sin(np.pi*x)*(1-x**2)
fe = ue.diff(x, 2)

# Lambdify for faster evaluation
ul = lambdify(x, ue, 'numpy') 
fl = lambdify(x, fe, 'numpy')

N = 32
SD = ShenDirichletBasis(N, plan=True)
X = SD.mesh(N)
u = TrialFunction(SD)
v = TestFunction(SD)
fj = fl(X)

# Compute right hand side of Poisson equation
f_hat = inner(v, fj) # array

# Get left hand side of Poisson equation and solve
A = inner(v, div(grad(u))) # matrix 
f_hat = A.solve(f_hat)
uj = SD.backward(f_hat)

# Compare with analytical solution
ue = ul(X)
assert np.allclose(uj, ue)



