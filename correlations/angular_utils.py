import h5py
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_data():
    print('Loading data')
    with h5py.File('data.h5', 'r') as f:
        phi1_f = f['phi1'][:]
        phi2_f = f['phi2'][:]
        v0_f = f['delta0'][:]
        v1_f = f['delta1'][:]
        v2_f = f['delta2'][:]

    return phi1_f, phi2_f, v0_f, v1_f, v2_f


def save_data(phi1_f, phi2_f, v0_f, v1_f, v2_f):
    print('Saving data')
    with h5py.File('data.h5', 'w') as f:
        f['phi1'] = phi1_f.astype(np.float32)
        f['phi2'] = phi2_f.astype(np.float32)
        f['delta0'] = v0_f.astype(np.float32)
        f['delta1'] = v1_f.astype(np.float32)
        f['delta2'] = v2_f.astype(np.float32)


def plot_histogram(phi1_f, phi2_f, v0_f, v1_f, v2_f):
    print('Plotting histogram')
    pi = np.pi
    plt.cla()
    bins = np.linspace(0, 2*np.pi, 100) / np.pi
    kwa = dict(alpha=0.5, normed=True, bins=bins)
    phi1_bin, phi1_edges, _ = plt.hist(phi1_f/pi,
                                       label=r'$\phi_1$', **kwa)
    phi2_bin, phi2_edges, _ = plt.hist(phi2_f/pi,
                                       label=r'$\phi_2$', **kwa)
    plt.xticks([0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2],
               ["0", "$\pi/4$", "$\pi/2$", "$3\pi/4$", "$\pi$",
               "$5\pi/4$", "$3\pi/2$", "$7\pi/4$", "$2\pi$"])
    plt.legend()
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$p(\phi)$')
    plt.grid('on')
    plt.savefig('hist_phi.pdf')

    phi1_max = phi1_edges[np.argmax(phi1_bin)]
    phi2_max = phi2_edges[np.argmax(phi2_bin)]
    print(f'phi_1.max == {phi1_max:.2f} pi')
    print(f'phi_2.max == {phi2_max:.2f} pi')


def plot_surface(phi1_f, phi2_f, v0_f, v1_f, v2_f):
    from mpl_toolkits.mplot3d import Axes3D
    from scipy.stats import binned_statistic_2d
    import matplotlib.cm as cm
    from matplotlib.cm import bwr as cmap
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    H, xedge, yedge = np.histogram2d(phi1_f, phi2_f, bins=50)
    X, Y = np.meshgrid(xedge[1:], yedge[1:])
    deltamean, _, _, _ = binned_statistic_2d(phi1_f, phi2_f, v1_f,
                                             bins=100, statistic='mean')

    color = cmap(deltamean.T)
    ax.plot_surface(X, Y, H.T, linewidth=0, facecolors=color)
    vmin, vmax = deltamean.min(), deltamean.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    ax1 = fig.add_axes([0.8, 0.1, 0.05, .8])
    cb = mpl.colorbar.ColorbarBase(
        ax1, cmap=cmap,
        norm=norm,
        orientation='vertical',
    )
    cb.set_label(r'$\left\langle \delta_1 \right\rangle$')


def plot_all(phi1_f, phi2_f, v0_f, v1_f, v2_f):
    plot_histogram(phi1_f, phi2_f, v0_f, v1_f, v2_f)
    plot_surface(phi1_f, phi2_f, v0_f, v1_f, v2_f)
