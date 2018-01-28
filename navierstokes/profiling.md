```
(shenfun) br222-003:navierstokes aponte$ kernprof -lv NavierStokes.py
Wrote profile results to NavierStokes.py.lprof
Timer unit: 1e-06 s

Total time: 0.006352 s
File: NavierStokes.py
Function: LinearRHS at line 46

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    46                                           @profile
    47                                           def LinearRHS():
    48         2       1767.0    883.5     27.8      A = inner(u, v)
    49         2       4582.0   2291.0     72.1      L = inner(nu*div(grad(u)), v) / A  # L is shape (N[0], N[1], N[2]//2+1), but used as (3, N[0], N[1], N[2]//2+1) due to broadcasting
    50                                               #L = -nu*K2  # Or just simply this
    51         2          3.0      1.5      0.0      return L

Total time: 1.37453 s
File: NavierStokes.py
Function: NonlinearRHS at line 53

Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    53                                           @profile
    54                                           def NonlinearRHS(U, U_hat, dU):
    55                                               global TV, curl_hat, curl_, P_hat, K, K_over_K2
    56        80       8953.0    111.9      0.7      dU.fill(0)
    57        80     158247.0   1978.1     11.5      U = TV.backward(U_hat, U)
    58        80     741883.0   9273.5     54.0      curl_hat = project(curl(U), TV, output_array=curl_hat, uh_hat=U_hat) # Linear. Does not actually use U, only U_hat
    59        80     144682.0   1808.5     10.5      curl_ = TV.backward(curl_hat, curl_)
    60        80      72662.0    908.3      5.3      W = np.cross(U, curl_, axis=0)                  # Nonlinear term in physical space
    61        80     171633.0   2145.4     12.5      dU = project(W, TV, output_array=dU)            # dU = TV.forward(W, dU)
    62        80      48244.0    603.0      3.5      P_hat = np.sum(dU*K_over_K2, 0, out=P_hat)
    63        80      28109.0    351.4      2.0      dU -= P_hat*K
    64        80        122.0      1.5      0.0      return dU
```
