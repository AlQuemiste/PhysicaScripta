import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def hamiltonianA(Lx:int, Ly:int, eAA:float, tAA:float):
    NN = 2 * Lx * Ly
    H = np.zeros((NN, NN), dtype=float)

    for i in range(Ly):  # row = y
        for j in range(Lx):  # column = x
            idxA0 = 2 * (i * Lx + j)  # A(i, j)

            # nearest neighbours with periodic boundary conditions
            jR = j + 1  # right neighbour
            if jR >= Lx:
                jR = 0

            iD = i + 1  # down neighbour
            if iD >= Ly:
                iD = 0

            idxAR = 2 * (i * Lx + jR)  # A(i, j + 1)
            idxAD = 2 * (iD * Lx + j)  # A(i + 1, j)

            # tAA c_Ai^+ c_Aj + h.c., for nearest-neighbour i and j
            H[idxA0, idxAR] = H[idxAR, idxA0] = -tAA
            H[idxA0, idxAD] = H[idxAD, idxA0] = -tAA

            # EAA c_Ai^+ c_Ai, local energy
            H[idxA0, idxA0] = eAA

    return H


def hamiltonianB(Lx:int, Ly:int, eBB:float, tBB:float):
    NN = 2 * Lx * Ly
    H = np.zeros((NN, NN), dtype=float)

    for i in range(Ly):  # row = y
        for j in range(Lx):  # column = x
            idxB0 = 2 * (i * Lx + j) + 1  # B(i, j)

            # nearest neighbours with periodic boundary conditions
            jR = j + 1  # right neighbour
            if jR >= Lx:
                jR = 0

            iD = i + 1  # down neighbour
            if iD >= Ly:
                iD = 0

            idxBR = 2 * (i * Lx + jR) + 1  # B(i, j + 1)
            idxBD = 2 * (iD * Lx + j) + 1  # B(i + 1, j)

            # tBB c_Bi^+ c_Bj + h.c., for nearest-neighbour i and j
            H[idxB0, idxBR] = H[idxBR, idxB0] = -tBB
            H[idxB0, idxBD] = H[idxBD, idxB0] = -tBB

            # EBB c_Bi^+ c_Bi, local energy
            H[idxB0, idxB0] = eBB

    return H


def hamiltonianAB(Lx:int, Ly:int, tAB:float):
    NN = 2 * Lx * Ly
    H = np.zeros((NN, NN), dtype=float)

    for i in range(Ly):  # row = y
        for j in range(Lx):  # column = x
            idxA0 = 2 * (i * Lx + j)      # A(i, j)
            idxB0 = 2 * (i * Lx + j) + 1  # B(i, j)

            # nearest neighbours with periodic boundary conditions
            iU = i - 1  # upper dark neighbour
            if iU < 0:
                iU = Ly - 1

            idxBU = 2 * (iU * Lx + j) + 1  # B(i - 1, j)

            # tBB c_Bi^+ c_Bj + h.c., for nearest-neighbour i and j
            H[idxA0, idxB0] = H[idxB0, idxA0] = -tAB
            H[idxA0, idxBU] = H[idxBU, idxA0] = -tAB

    return H


def solve(H:'matrix'):
    """ solve the eigenvalue problem, H \psi = E \psi """

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def lattice_interplot(sites:[(float, float)], values:[float],
                      points:int=32,
                      vmin=None, vmax=None, cmap='viridis'):
    """ plot values over given lattice sites using interpolation """

    fig_width = 8
    fig, ax1 = plt.subplots(nrows=1, ncols=1)
    fig.set_figwidth(fig_width)  # width of the figure in inches

    Xs, Ys = sites[:, 0], sites[:, 1]
    x_min, x_max = Xs.min(), Xs.max()
    y_min, y_max = Ys.min(), Ys.max()

    # interpolate the data
    x_p = np.linspace(x_min, x_max, points)
    y_p = np.linspace(y_min, y_max, points)
    xx, yy = np.meshgrid(x_p, y_p)
    z_p = griddata(sites, values, (xx, yy), method='cubic',
                   fill_value=0, rescale=False)

    intrp_plt = ax1.imshow(z_p, extent=(x_min, x_max, y_min, y_max),
                           vmin=vmin, vmax=vmax,
                           cmap=cmap, origin='lower')
    cbar = fig.colorbar(intrp_plt)
    return fig, ax1


def sites(Lx:int, Ly:int):
    # coordinates of lattice sites
    sites = list()

    for i in range(Ly):  # row = y
        for j in range(Lx):  # column = x
            # A(i, j): bright
            xA = 2 * j
            yA = 2 * i
            # B(i, j): dark
            xB = xA + 1
            yB = yA + 1

            sites.extend([[xA, yA], [xB, yB]])

    return sites


def plotDoS(energyLevels):
    """ plot density of states (energy levels and their line-widths) """

    # plot
    fig, dos_plt = plt.subplots(1, 1, dpi=300, clear=True)

    _ymax = 1.0
    for energy in energyLevels:
        dos_plt.vlines(energy, 0, _ymax, linestyles='solid',
                       linewidth=0.5, color='gray')

    dos_plt.set_ylim(bottom = 0)

    return fig, dos_plt


def histogram(data, n_bins):
    """
    create a histogram from data with a specified number of bins
    """

    # data range
    data_min, data_max = np.min(data), np.max(data)
    # bin edges
    bin_edges = np.linspace(data_min, data_max, n_bins + 1)
    # histogram
    hist, edges = np.histogram(data, bins=bin_edges)

    # plot
    fig, dos_plt = plt.subplots(1, 1, dpi=300, clear=True)
    dos_plt.hist(data, bins=bin_edges, color='orangered', density=True)
    dos_plt.set_title(f'DoS with {n_bins} bins')
    dos_plt.set_xlabel('Energy')
    dos_plt.set_ylabel('DoS')

    return fig, dos_plt, edges, hist


class LDoSPlot:
    """ Plot LDoS """

    def __init__(self, n_sites, energyLevels, eigenStates, width):
        self.energyLevels = energyLevels
        self.eigenStates = eigenStates
        self.width = width
        self.n_sites = n_sites

        self._nEnergies = self.energyLevels.size
        self._norm_coef = 1.0 / (np.pi * self._nEnergies)

    def _lorentz_E(self, E_0, E):
        """ Lorentz peak """

        w_2 = self.width**2
        c0 = self._norm_coef * self.width
        return c0 / ((E - E_0)**2 + w_2)

    def LDoS_E(self, E):
        """ calculate LDoS for a given energy """
        n0 = 1 / self.n_sites
        ldos_E = n0 * np.sum(self.eigenStates**2 * self._lorentz_E(self.energyLevels, E),
                             axis=1)
        return ldos_E


def plotResults(datafile:str, out_path:str, states=False):
    data = np.load(datafile)
    Lx = int(data['Lx'])
    Ly = int(data['Ly'])
    a = int(data['a'])
    V0 = float(data['V0'])
    V = data['V']
    evals = data['eigenvalues']
    evecs = data['eigenvectors']

    Nx = Lx * a
    Ny = Ly * a
    sites = np.array([(j, i) for j in range(0, Ny) for i in range(0, Nx)])

    # plot the potential
    figV, ax1 = lattice_interplot(sites, V, points=128, cmap='viridis')
    ax1.set_title("Potential V(x,y)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plt.savefig(out_path + "/potential.png", dpi=150, format='png')
    plt.clf()

    if not states:
        return

    print("* Plotting eigenstates (output in '%s')..." % out_path)

    n_states = evals.size
    fn_fmt = out_path + "/state_{:0" + str(int(np.log10(n_states))+1) + "d}.png"

    for i_e in range(n_states):
        s_i = evecs[:, i_e]
        fig, ax1 = lattice_interplot(sites, np.abs(s_i)**2, points=128, cmap='plasma')
        ax1.set_title(f'Eigenstate {i_e} with E = {evals[i_e]:.3f}')
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        plt.savefig(fn_fmt.format(i_e), dpi=150, format='png')
        plt.close()

    print("* Plotting eigenstates: done.")


def solveTB(Lx:int, Ly:int,
            eAA:float, tAA:float,
            eBB:float, tBB:float,
            tAB:float, data_path:str):

    print("* Constructing the Hamiltonian matrix...")
    H_tb = hamiltonianA(Lx, Ly, eAA, tAA) \
        + hamiltonianB(Lx, Ly, eBB, tBB) \
        + hamiltonianAB(Lx, Ly, tAB)

    print("* Constructing the Hamiltonian matrix: done. shape =", H_tb.shape)

    print("* Solving the eigenproblem...")
    evals, evecs = solve(H_tb)
    print("* Solving the eigenproblem: done.")

    # save results
    datafile = data_path + "/eigensystem.npz"
    np.savez(datafile, hamiltonian=H_tb,
             Lx=Lx, Ly=Ly,
             eAA=eAA, tAA=tAA,
             eBB=eBB, tBB=tBB, tAB=tAB,
             eigenvalues=evals, eigenvectors=evecs)

    print("* Results stored in '%s'." % datafile)
    return datafile

#----------------------------------------

if __name__ == "__main__":
    import os
    out_path = data_path = "/tmp/results"
    os.makedirs(data_path, exist_ok=True)

    Lx = 2**6
    Ly = 2**6
    eAA = 1
    eBB = 2 * eAA
    tAA = 0.1
    tBB = tAA
    tAB = 0.5 * tAA

    datafile = solveTB(Lx, Ly, eAA, tAA, eBB, tBB, tAB, data_path)

    # read data
    # data = np.load(datafile)
    # Lx = int(data['Lx'])
    # Ly = int(data['Ly'])
    # evals = data['eigenvalues']
    # evecs = data['eigenvectors']

    # plotResults(datafile, out_path)
