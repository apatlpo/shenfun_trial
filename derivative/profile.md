```
(shenfun) br222-003:derivative aponte$ kernprof -lv fourier.py
Wrote profile results to fourier.py.lprof
Timer unit: 1e-06 s

Total time: 0.18519 s
File: fourier.py
Function: Dh at line 56

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    56                                               @profile
    57                                               def Dh(self, dvar, dim):
    58                                                   ''' Wrapper around Dx
    59                                                   '''
    60       100     185007.0   1850.1     99.9          dvar[:] = self.T.backward(project(Dx(self.hf, dim, 1), self.T))
    61       100        183.0      1.8      0.1          return dvar

Total time: 0.038293 s
File: fourier.py
Function: Dh at line 71

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    71                                               @profile
    72                                               def Dh(self,dvar,dim):
    73                                                   ''' Wrapper around Dx
    74                                                   '''
    75       100      18365.0    183.7     48.0          self.work = self.T.forward(self.h, self.work)
    76       100      19836.0    198.4     51.8          dvar = self.T.backward((1j*self.K[dim])*self.work, dvar)
    77       100         92.0      0.9      0.2          return dvar

Total time: 0.037733 s
File: fourier.py
Function: Dh at line 92

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    92                                               @profile
    93                                               def Dh(self, dvar, dim):
    94                                                   ''' Wrapper around Dx
    95                                                   '''
    96       100      18515.0    185.2     49.1          self.work1 = self.T.forward(self.h, self.work1)
    97       100       3142.0     31.4      8.3          deriv(self.work1, self.work2, np.squeeze(self.K[0]), np.squeeze(self.K[1]), dim)
    98       100      15990.0    159.9     42.4          dvar = self.T.backward(self.work2, dvar)
    99       100         86.0      0.9      0.2          return dvar
```
