
```
(shenfun) barracuda:derivative aponte$ python -m line_profiler fourier.py.lprof
Timer unit: 1e-06 s

Total time: 0.161684 s
File: fourier.py
Function: Dh at line 56

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    56                                               @profile
    57                                               def Dh(self, dvar, dim):
    58                                                   ''' Wrapper around Dx
    59                                                   '''
    60       100     161545.0   1615.5     99.9          dvar[:] = self.T.backward(project(Dx(self.hf, dim, 1), self.T))
    61       100        139.0      1.4      0.1          return dvar

Total time: 0.03875 s
File: fourier.py
Function: Dh at line 71

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    71                                               @profile
    72                                               def Dh(self,dvar,dim):
    73                                                   ''' Wrapper around Dx
    74                                                   '''
    75       100      18540.0    185.4     47.8          self.work = self.T.forward(self.h, self.work)
    76                                                   #dvar = self.T.backward((1j*self.K[dim])*self.work, dvar)
    77       100      20122.0    201.2     51.9          dvar = self.T.backward(Kmult(self.K[dim],self.work), dvar)
    78       100         88.0      0.9      0.2          return dvar

Total time: 0.004038 s
File: fourier.py
Function: Kmult at line 80

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    80                                           @profile
    81                                           def Kmult(K,work):
    82       100       4038.0     40.4    100.0      return (1j*K)*work

Total time: 0.031006 s
File: fourier.py
Function: Dh at line 96

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    96                                               @profile
    97                                               def Dh(self, dvar, dim):
    98                                                   ''' Wrapper around Dx
    99                                                   '''
   100       100      15402.0    154.0     49.7          self.work1 = self.T.forward(self.h, self.work1)
   101       100       2713.0     27.1      8.7          deriv(self.work1, self.work2, np.squeeze(self.K[0]), np.squeeze(self.K[1]), dim)
   102       100      12817.0    128.2     41.3          dvar = self.T.backward(self.work2, dvar)
   103       100         74.0      0.7      0.2          return dvar
```

