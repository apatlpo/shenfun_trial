
# Notes for shenfun paper:

1. `For example the stiffness matrix K`, I suspect not all reader will remember what linear and bilinear forms are. It may be useful to remind it somewhere and  precise in this particular example that it is a linear form (right?)

2. I would use bhat and uhat in this example (and the following one).
It makes it clearer that we are in spectral space.
```
>>> fj = np.random.random(N)
>>> b = inner(v, fj)
>>> u = np.zeros_like(b)
>>> u = K.solve(b, u)
```

3.  I've got the following issue:
```
In [18]: fj = np.random.random((N, N, N))
    ...: bhat = inner(v, fj)
    ...:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
<ipython-input-18-2c4a59c93951> in <module>()
      1 fj = np.random.random((N, N, N))
----> 2 bhat = inner(v, fj)

~/.miniconda3/envs/shenfun/lib/python3.6/site-packages/shenfun/forms/inner.py in inner(expr0, expr1, output_array, uh_hat)
    109         space = test.function_space()
    110         if isinstance(trial, np.ndarray):
--> 111             output_array = space.scalar_product(trial, output_array)
    112             return output_array
    113

~/.miniconda3/envs/shenfun/lib/python3.6/site-packages/shenfun/spectralbase.py in __call__(self, input_array, output_array, **kw)
    576     def __call__(self, input_array=None, output_array=None, **kw):
    577         if input_array is not None:
--> 578             self.input_array[...] = input_array
    579         self.func(None, None, **kw)
    580         if output_array is not None:

ValueError: could not broadcast input array from shape (8,8,8) into shape (8)
```

4. It may be helpful to mention explicitely and early that inner may output an array (equivalent to a forward transform) or a matrix. This is a crucial (and very nice) aspect of the library but it took me a while to figure it out.

5. Poisson1D example with Fourier approach, add backward transform:
```
f_hat = ST.forward(fj)
# Wavenumers
k = ST.wavenumbers(N)
k[0] = 1
# Solve Poisson equation (solution in f_hat)
f_hat /= k**2
uq_fourier = ST.backward(f_hat)
```

6. Make poisson1D with chebyshev approach self contained with imports
```
from sympy import Symbol , sin, lambdify
import numpy as np
from shenfun import inner , div , grad , TestFunction , TrialFunctio
from shenfun.chebyshev.bases import ShenDirichletBasis
```

7. Add missing libraries for TensorProductSpace illustration, and maybe use T instead of W for the tensor product space in order to be consistent with poisson2D example:
```
from shenfun.chebyshev.bases import ShenDirichletBasis
from shenfun.fourier.bases import FourierBasis
import numpy as np
```

8. Add command to run with mpi4py: `mpirun -n 4 python poisson2D.py`

9. `Function has a keyword argument forward_output, that is used as w_hat
= Function(W, forward_output=True)` mention `forward_output=True` is default

10. `The main difference is that for Legendre it is natural to integrate the weak form by parts and use`. This is not straightfoward. Reference or explanation maybe useful.

11. Add missing libraries to poisson2D example. T is not consistent with W in preceding illustration of tensor product space. For example:
```
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

...
```

12. While I loosely guess what Helmholtz problem and solvers refers to, it may
be useful to define or explain this more clearly. 

13. 3D example has the W vs T ambiguity

14. A sketch of the MPI domain decomposition, physical vs spectral ones may be more efficient at describing how this is done.

15. For the biharmonic problem, it is no clear why dimension 1 and 2 are different:
```
K1 = C2CBasis(N[1])
K2 = R2CBasis(N[2])
```

16. Some imports are missing in biharmonic pb, fj does not exist:
```
from shenfun.chebyshev.bases import ShenBiharmonicBasis
from shenfun.fourier.bases import R2CBasis, C2CBasis
from shenfun.chebyshev.la import Biharmonic as Solver
from shenfun import inner , div , grad , TestFunction , TrialFunction
from shenfun import Function , TensorProductSpace
from mpi4py import MPI
comm = MPI.COMM_WORLD
...
fj = ...
```


