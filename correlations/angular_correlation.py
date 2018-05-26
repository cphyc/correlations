import numpy as np
from utils import W1G
from scipy.special import spherical_jn as j
from numpy import sin, cos, sqrt

k, Pk, _ = np.loadtxt('power.dat', skiprows=1).T.astype(np.float32)
Pk *= 2*np.pi**2 * 4*np.pi

R = 5   # Mpc/h

phi = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
pos = R * np.array([cos(phi), sin(phi), np.zeros_like(phi)])
r = sqrt(np.sum((pos[:, None, :] - pos[:, :, None])**2, axis=0))
r[r == 0] = 1e-4

phi1 = phi[:, None]
phi2 = phi[None, :]
phimphi = phi2 - phi1
phimphi[phimphi == 0] = 1e-4

###############################################################################
# Compute base elements
###############################################################################
k2Pk = k[None, None, :]**2 * Pk[None, None, :]
kR = k[None, None, :] * R
kr = k[None, None, :] * r[:, :, None]
WW = W1G(kR) * W1G(kR)
xifun = lambda n, m: np.trapz(k2Pk * WW * j(n, kr) / (kr)**m / (2*np.pi**2), k)
chifun = lambda n, m: np.trapz(k2Pk * WW * j(n, kr) * (kr)**m / (2*np.pi**2), k)

# The limits at 0 distance are all null
xi00 = np.where(r == 0, 0, xifun(0, 0))
xi00p = np.where(r == 0, 0, -1 / r * chifun(1, 1))
xi00pp = 1 / r**2 * (chifun(2, 2) - chifun(1, 1))
xi00ppp = 1 / r**3 * (2 * chifun(0, 2) - 6*chifun(1, 1) + chifun(1, 3))


###############################################################################
# Compute correlation coefficients
###############################################################################
deltadelta = xi00
tmp = np.where(phimphi == 0, 0, sin(phimphi) * xi00p / r)
deltagrad = np.array([phimphi*0, tmp])

# grad-grad term
A = -xi00p / r
B = sqrt(2) * sin(phimphi / 2)**4 * xi00p / (R * (1-cos(phimphi))**(3/2)) \
    - cos(phimphi / 2)**2 * xi00pp
_0 = np.zeros_like(A)
gradgrad = np.array([[A, _0], [_0, B]])

# grad-Hessian term
_0 = np.zeros_like(phi1 + phi2)
gradhess = 1 / r**2 * np.array([
    [
        [
            _0,
            1/2 * sin(phi2)**2 * sin(phi1+phi2) * (sqrt(2) * (2 + cos(phi1 + phi2)) * xi00p / (sqrt(1 + cos(phi1 + phi2))) + 2 * R + cos(phi1 + phi2) * xi00pp)
        ], [
            1/2 * sin(phi1) * sin(phi2) * sin(phi1 + phi2) * (-sqrt(2)*xi00p / sqrt(1 + cos(phi1 + phi2)) + 2 * R * xi00pp),
            _0
        ]
    ], [
        [
            1/2 * sin(phi1) * sin(phi2) * sin(phi1 + phi2) * (-sqrt(2)*xi00p / sqrt(1 + cos(phi1 + phi2)) + 2 * R * xi00pp),
            _0
        ], [
            _0,
            2 * cos((phi1 + phi2) / 2)**5 * sin((phi1 + phi2) / 2) / (1 + cos(phi1 + phi2))**(3/2) * (sqrt(2) * xi00p + 2*R*(3*sqrt(1 + cos(phi1+phi2)) * xi00pp + sqrt(2) * R * (-1 + cos(phi1+phi2)) * xi00ppp))
        ]
    ]
])

# # Hessian-Hessian term
# hesshess = np.zeros()
