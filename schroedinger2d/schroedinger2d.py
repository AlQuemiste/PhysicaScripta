import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


def periodicPotential(x:int, y:int, a:int, V0:float, sigma:float=None):
    """
    periodic 2d potential V(x,y)
    V(x+a, y) = V(x, y+a) = V(x,y)
    """
    c0 = np.pi / a

    if sigma is None:
        sigma = 0.5 * a

    g_tip = 1 / (5 * sigma)
    x_tip = 30
    y_tip = 40
    V_tip = 0.01 * V0 * (np.exp( -(g_tip * (x - x_tip))**2 ) \
                         * np.exp( -(g_tip * (y - y_tip))**2 ))

    g0 = 1.0 / (2 * sigma)
    V_lattice = -V0 * (np.exp( -(g0 * np.cos(c0 * x))**2 ) * np.exp( -(g0 * np.cos(c0 * y))**2 ))
    # V0 * (np.cos(c0 * x) + np.cos(c0 * y))
    return V_lattice + V_tip


def hamiltonian(Lx:int, Ly:int, a:int, V0:float):
    Nx = Lx * a
    Ny = Ly * a
    NN = Nx * Ny
    H = np.zeros((NN, NN), dtype=float)
    V = np.zeros(NN, dtype=float)

    # - d^2 F / dx^2 - d^2 F / dy^2 + V(x,y)
    for i in range(Ny):  # row = y
        for j in range(Nx):  # column = x
            idx0 = i * Nx + j   # (i, j)

            if j + 1 < Nx:
                idx_xf = idx0 + 1   # (i, j + 1)
                H[idx0, idx_xf] += -1
            if j - 1 >= 0:
                idx_xb = idx0 - 1   # (i, j - 1)
                H[idx0, idx_xb] += -1
            if i + 1 < Ny:
                idx_yf = idx0 + Nx  # (i + 1, j)
                H[idx0, idx_yf] += -1
            if i - 1 >= 0:
                idx_yb = idx0 - Nx  # (i - 1, j)
                H[idx0, idx_yb] += -1

            H[idx0, idx0] += -4

            # potential
            v_xy = V[idx0] = periodicPotential(j, i, a, V0)
            H[idx0, idx0] += v_xy

    return H, V, Nx, Ny


def solve(H:'matrix'):
    """ solve the eigenvalue problem, H \psi = E \psi """

    eigenvalues, eigenvectors = np.linalg.eigh(H)
    return eigenvalues, eigenvectors


def evalPotential(x:'1d array', y:'1d array', a:int, V0:float):
    # evaluate the potential on the given grid
    X, Y = np.meshgrid(x, y)
    V = periodicPotential(X, Y, a, V0)
    return X, Y, V


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


def solveSchroedinger(Lx:int, Ly:int, a:int, V0:float, data_path:str):
    print("* Constructing the Hamiltonian matrix...")
    H0, V, Nx, Ny = hamiltonian(Lx, Ly, a, V0)
    print("* Constructing the Hamiltonian matrix: done. shape =", H0.shape)

    print("* Solving the eigenproblem...")
    evals, evecs = solve(H0)
    print("* Solving the eigenproblem: done.")

    # save results
    datafile = data_path + "/eigensystem.npz"
    np.savez(datafile, hamiltonian=H0,
             Lx=Lx, Ly=Ly, a=a, V0=V0, Nx=Nx, Ny=Ny,
             V=V, eigenvalues=evals, eigenvectors=evecs)

    print("* Results stored in '%s'." % datafile)
    return datafile

#----------------------------------------

if __name__ == "__main__":
    import os
    out_path = data_path = "/tmp/results"
    os.makedirs(data_path, exist_ok=True)

    Lx = 2**4
    Ly = 2**4
    V0 = 5
    a = 4

    datafile = solveSchroedinger(Lx, Ly, a, V0, data_path)

    plotResults(datafile, out_path, states=True)
