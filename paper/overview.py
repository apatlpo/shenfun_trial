
import shenfun
from shenfun import inner, TestFunction, TrialFunction, div, grad
import numpy as np

N = 8

B = shenfun.chebyshev.bases.ShenDirichletBasis(N, plan=True)
#B = FFT = shenfun.fourier.bases.R2CBasis(N, plan=True)

u = TrialFunction(B)
v = TestFunction(B)
mass = inner(u, v) # spectral matrix, i.e. a dict
print(type(mass)) 
print(mass)

# linear and bilinear form
K = inner(v, div(grad(u)))

# solves Ku=b
fj = np.random.random(N)
bhat = inner(v, fj) # transform of fj, same as B.forward(fj) 
#bhat2 = B.forward(fj) 
print(type(mass)) 
arint(type(mass)) 
uhat = np.zeros_like(bhat)
uhat = K.solve(bhat, uhat) # solution in spectral space
u = B.backward(uhat) # back to physical space

# on a three dimensional cube, t
# the matrix solve is applied along the first dimension since this is the default behaviour.
fj = np.random.random((N, N, N))
bhat = inner(v, fj)
uhat = np.zeros_like(bhat)
uhat = K.solve(bhat, uhat)

# transform back and forth
fj = np.random.random(N) # does not satisfy the basis bdy conditions
fj0 = fj.copy()
fk = np.zeros_like(fj)
fk = B.forward(fj, fk) # Gets expansion coefficients
fj = B.backward(fk, fj) # not the original fj because of bdy conditions
print(fj)
print(fj0-fj)

fj1 = fj.copy()
fk = B.forward(fj, fk)
fj = B.backward(fk, fj)
print(np.allclose(fj, fj1))

print(fj)
print(fj1-fj)


