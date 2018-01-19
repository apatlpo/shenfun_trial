
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




