```
(shenfun) br222-003:navierstokes aponte$ kernprof -lv NavierStokes.py
Wrote profile results to NavierStokes.py.lprof
Timer unit: 1e-06 s

Total time: 0.484557 s
File: NavierStokes.py
Function: NonlinearRHS at line 53

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    53                                           @profile
    54                                           def NonlinearRHS(U, U_hat, dU):
    55                                               global TV, curl_hat, curl_, P_hat, K, K_over_K2
    56        40       1995.0     49.9      0.4      dU.fill(0)
    57        40      52641.0   1316.0     10.9      U = TV.backward(U_hat, U)
    58        40     263274.0   6581.9     54.3      curl_hat = project(curl(U), TV, output_array=curl_hat, uh_hat=U_hat) # Linear. Does not actually use U, only U_hat
    59        40      48995.0   1224.9     10.1      curl_ = TV.backward(curl_hat, curl_)
    60        40      26993.0    674.8      5.6      W = np.cross(U, curl_, axis=0)                  # Nonlinear term in physical space
    61        40      63221.0   1580.5     13.0      dU = project(W, TV, output_array=dU)            # dU = TV.forward(W, dU)
    62        40      15859.0    396.5      3.3      P_hat = np.sum(dU*K_over_K2, 0, out=P_hat)
    63        40      11543.0    288.6      2.4      dU -= P_hat*K
    64        40         36.0      0.9      0.0      return dU
```
