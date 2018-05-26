import numpy as np
from correlations.utils import Utils
from scipy.integrate import dblquad
from time import time

pi = np.pi

k = np.geomspace(1e-4, 1e4, 3000)
Pk = k**-2

u = Utils(k, Pk)

print('%10s%14.5e' % ('sigma=', u.sigma(0, 8)))
(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2) = 0, 0, 0, 0, 1, 0, 0, 1, 1

for ikx in [0, 1, 2]:
    for ikk in [0, 2]:
        before = time()
        res = dblquad(
            u.integrand_lambdaCDM,
            0, pi,                      # theta bounds
            lambda theta: 0, lambda theta: 2*pi,  # phi bounds
            epsrel=1e-3, epsabs=1e-5,
            args=(ikx, iky, ikz, ikk, dx, dy, dz, R1, R2))[0]
        now = time()
        print('ikx=%2d ikk=%2d  =%14.5e t=%6.2f mu/call' % (ikx, ikk, res, (now-before)*1000))
