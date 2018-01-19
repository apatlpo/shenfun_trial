from shenfun.chebyshev.bases import ShenDirichletBasis
from shenfun.fourier.bases import FourierBasis
from shenfun import Function , TensorProductSpace
# TensorProductSpace class is used to construct W , 
# Function is a subclass of numpy.ndarray used to hold solution arrays.
from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
N = (32, 33)

K0 = ShenDirichletBasis(N[0])
K1 = FourierBasis(N[1], dtype=np.float)
W = TensorProductSpace(comm, (K0, K1))
print(W)

# Alternatively, switch order for periodic in first direction instead
# W = TensorProductSpace(comm, (K1, K0), axes=(1, 0))

#
# w_hat = Function(W, forward_output=True) 
#	to create an array consistent with the output of W.forward (solution in spectral space)
# w = Function(W, forward_output=False) 
#	to create an array consistent with the input (solution in real space).

#
# uh = np.zeros_like(w_hat)
# w_hat = Function(W, buffer=uh) 
#  can be used to wrap a Function instance around a regular Numpy array uh. 
#  Note that uh and w_hat now will share the same data, and modifying one will 
#  naturally modify also the other.


