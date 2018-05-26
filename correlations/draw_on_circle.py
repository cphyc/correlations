import numpy as np
from scipy.special import spherical_jn as bessj
from tqdm import tqdm
import os

import angular_utils as tu
import utils as U
from utils import W1TH, W1G, trapz, get_maxis

###############################################################################
# Parameters
###############################################################################
# Radius of circle
Rc = 10      # radius of circle, in Mpc/h
R  = 5       # smoothing scale, in Mpc/h
Rcenter = 5  # smoothing scale at center, in Mpc/h
Rwalls = 5   # smoothing scale at walls, in Mpc/h
Npts = 100
epsilon = 1e-5

k, Pk, _ = np.loadtxt('power.dat', skiprows=1).T.astype(np.float32)
Pk *= 2*np.pi**2 * 4*np.pi


###############################################################################
# Compute the covariance matrix
###############################################################################
# Draw positions on the circle
phi = np.concatenate((np.linspace(0, 2*np.pi, Npts, dtype=np.float32),
                      [0, 0, 2*np.pi/3, -2*np.pi/3]))
N = len(phi)
Nconstrains = len(phi) - Npts

theta = np.ones_like(phi) * np.pi/2
Radii = np.ones_like(phi) * Rc

# Index of particular points
iCenter = len(phi) - 4
iWalls  =len(phi) - 3
Radii[iCenter] = 0
Radii[iWalls:] = 2 * Rc

pos = Radii * np.array([np.sin(theta) * np.cos(phi),
                        np.sin(theta) * np.sin(phi),
                        np.cos(theta)])

# Compute 2-point distances
dist = np.sqrt(np.sum((pos[:, :, None] - pos[:, None, :])**2, axis=0))
dist[dist == 0] = epsilon
dx = pos[:, None, :] - pos[:, :, None]
graddist = dx / dist
diag = np.diag([1, 1, 1])[:, :, None, None] / dist
d2dist = -(dx[None, :, :, :]*dx[:, None, :, :]) / dist**3 + diag
# Set 0s for hessian of 0 separation terms
for i in range(3):
    for j in range(pos.shape[1]):
        d2dist[i, i, j, j] = 0

# Array containing the smoothing scales
RR = np.ones_like(phi) * R
RR[iCenter] = Rcenter
RR[iWalls:] = Rwalls

# Compute the correlation delta-delta
print("Computing correlation coefficients")
kr = k[None, None, :] * dist[:, :, None]
k2Pk = k[None, None, :]**2 * Pk[None, None, :]

# Get Kernel
W1 = W1G
W1kR = W1(k[None, :] * RR[:, None])
W1kRW1kR = W1kR[:, None, :] * W1kR[None, :, :]

xifun = lambda n, m: trapz(k2Pk * W1kRW1kR * bessj(n, kr) / kr**m / (2*np.pi**2), k)
chifun = lambda n, m: trapz(k2Pk * W1kRW1kR * bessj(n, kr) * kr**m / (2*np.pi**2), k)
xi00 = xifun(0, 0)
xi00p = -1/dist * chifun(1, 1)
xi00pp = 1/dist**2 * (chifun(2, 2) - chifun(1, 1))
# xi11 = trapz(k2Pk * W1kRW1kR * bessj1kr / (2*np.pi**2), k)
# xi20 = trapz(k2Pk * W1kRW1kR * bessj(2, kr) / (2*np.pi**2), k)

# Density-density
dd = xi00
dgradd = (graddist * xi00p)[:, iCenter:, :]
# dhessd = (d2dist * xi00p + graddist[:, None, :, :] * graddist[None, :, :, :]*xi00pp)[:, :, iCenter:, :]
gradgrad = (-d2dist * xi00p -
            graddist[:, None, :, :] * graddist[None, :, :, :]*xi00pp)[:, :, iCenter:, iCenter:]

S = (Npts + Nconstrains) + 3 * Nconstrains # + 6 * Nconstrains (for hessian)
cov = np.zeros((S, S))

# density-density, density-grad
cov[:N, :N] = xi00
for i in range(Nconstrains):
    igrad = N + i*3
    # density-gradient
    cov[igrad:igrad+3, :N] = dgradd[:, i, :]

# Grad-grad
for i in range(Nconstrains):
    igrad = N + i * 3
    for j in range(i):
        jgrad = N + j * 3
        cov[igrad:igrad+3, jgrad:jgrad+3] = gradgrad[:, :, i, j]
    sigma2grad = np.trapz(
        k2Pk * W1(k * RR[iCenter + i])**2 / (2*np.pi**2), k)
    cov[igrad:igrad+3, igrad:igrad+3] = np.diag([1, 1, 1]) * sigma2grad / 3


# Symmetrize
for i in range(N):
    for j in range(i):
        cov[j, i] = cov[i, j]

###############################################################################
# Constrain to the value at center
###############################################################################
constr = np.empty_like(mean) * np.nan
constr[-1] = 1.69

# Constrain
mean, cov = U.constrain(mean, cov, constr)

# No constrain
# mean, cov = mean[:-1], cov[:-1, :-1]

###############################################################################
# Draw random sample
###############################################################################
phi1_f = np.empty(0)
phi2_f = np.empty(0)
v0_f, v1_f, v2_f = [np.empty(0) for _ in range(3)]
Ntot = int(1e6)
Nperdraw = Ntot
prog = tqdm(total=Ntot)
u, s, v = np.dual.svd(cov)


def multivariate_normal(size):
    '''
    Draw random number from a multivariate_normal distribution. This
    is the same as numpy's implementation (but the variables u, s, v
    are cached)'''
    shape = [size, len(mean)]
    x = np.random.standard_normal(size=shape)
    x = np.dot(x, np.sqrt(s)[:, None] * v)
    x += mean
    return x


np.random.seed(16091992)

try:
    # if os.path.exists('data.h5'):
    #     phi1_f, phi2_f, v0_f, v1_f, v2_f = tu.load_data()
    #     prog.update(len(phi1_f))

    while len(phi1_f) < Ntot:
        sample = multivariate_normal(Nperdraw).astype(np.float32)
        # sample = np.random.multivariate_normal(mean, cov, size=Nperdraw)
        m0, m1, m2, v0, v1, v2 = get_maxis(sample)
        mask = (m0 >= 0) & (v2 > 0)
        phi0 = phi[m0[mask]]
        phi1 = phi[m1[mask]]
        phi2 = phi[m2[mask]]
        v0 = v0[mask]
        v1 = v1[mask]
        v2 = v2[mask]

        phi1 -= phi0
        phi2 -= phi0

        phi1 = np.mod(phi1, 2*np.pi)
        phi2 = np.mod(phi2, 2*np.pi)

        phi2 = np.where(phi1 > np.pi, 2*np.pi-phi2, phi2)
        phi1 = np.where(phi1 > np.pi, 2*np.pi-phi1, phi1)

        phi1_f = np.concatenate([phi1_f, phi1])
        phi2_f = np.concatenate([phi2_f, phi2])
        v0_f = np.concatenate([v0_f, v0])
        v1_f = np.concatenate([v1_f, v1])
        v2_f = np.concatenate([v2_f, v2])
        prog.update(len(phi1))

finally:
    tu.save_data(phi1_f, phi2_f, v0_f, v1_f, v2_f)
    tu.plot_histogram(phi1_f, phi2_f, v0_f, v1_f, v2_f)
    tu.plot_surface(phi1_f, phi2_f, v0_f, v1_f, v2_f)
