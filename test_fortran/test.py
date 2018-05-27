import numpy as np
from correlations.utils import Utils
from correlations.correlations import Correlator
from scipy.integrate import dblquad
from time import time

pi = np.pi

# k = np.geomspace(1e-4, 1e4, 3000)
# Pk = k**-2

# u = Utils(k, Pk)

# print('%10s%14.5e' % ('sigma=', u.sigma(0, 8)))
# (ikx, iky, ikz, ikk, dx, dy, dz, R1, R2) = 0, 0, 0, 0, 1, 0, 0, 1, 1

# for ikx in [0, 1, 2]:
#     for iky in [0, 1, 2]:
#         for ikz in [0, 1, 2]:
#             for ikk in [0, 2]:
#                 before = time()
#                 res = dblquad(
#                     u.integrand_lambdaCDM,
#                     0, pi,                      # theta bounds
#                     lambda theta: 0, lambda theta: 2*pi,  # phi bounds
#                     epsrel=1e-3, epsabs=1e-5,
#                     args=(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2))[0]
#                 now = time()
#                 print('ikx=%2d iky=%2d ikz=%2d ikk=%2d =%14.5e t=%6.2f mu/call' %
#                       (ikx, iky, ikz, ikk, res, (now-before)*1000))


# c = Correlator(quiet=True)
# c.add_point([0, 0, 0], ['hessian'], 1)
# c.add_point([1, 0, 0], ['hessian'], 1)
# print(c.k.shape)
# before = time()
# c.cov
# after = time()
# for i in range(12):
#     print(('%10.5f'*12) % tuple(np.round(c.cov[i, :]*15, 5)))
# print('t= %6.2f ms' % ((after-before)))

c = Correlator(quiet=True)
c.add_point([0, 0, 0],
            ['potential', 'a', 'tide', 'grad_delta', 'hessian'],
            1)
print(c.cov)
