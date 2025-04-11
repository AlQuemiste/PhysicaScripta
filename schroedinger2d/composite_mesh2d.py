import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple


EPS = np.finfo(float).eps * 1e6

Limits = namedtuple("Limits", "x, y")


def is_zero(x:float, atol):
    return (abs(x) < atol)


def is_close(x:float, y:float, atol):
    return is_zero(x - y, atol)


def prune_points(points:[float], d_min:float):
    """ Discards the points which are too close
        to their neighbour on the left.
    """

    if points.size <= 2:
        return points.copy()

    # keep the 1st point
    pts_pruned = [points[0]]

    pt_last = pts_pruned[0]
    for pt in points[1:-1]:
        if is_close(pt, pt_last, d_min): continue
        pts_pruned.append(pt)
        pt_last = pt

    # keep the last point
    pt_last = pts_pruned[-1]
    if is_close(pt_last, points[-1], d_min):
        pts_pruned[-1] = points[-1]
    else:
        pts_pruned.append(points[-1])

    return np.asarray(pts_pruned, dtype=float)


def compose(meshes:'[1d-array]', d_min:float) -> '1d-array':
    """ Makes a composite 1d mesh from the given meshes """

    # keep the 1st element, drop any further element
    # where distance between two consecutive points (x_i - x_{i-1})
    # is less than the allowed value
    composite = prune_points(np.sort(np.concatenate(meshes)), d_min)
    return composite


def unitcell_mesh1d(a1:'1d-array', a2:'1d-array', basis:['1d-array'],
                    axis:int, n_mid:int, d_min:float):
    """ Produces a 1d mesh for a single unit cell of 2d lattice
        on a given axis.

        The origin of the unit cell is assumed to be at zero.
    """

    rO:float = 0  # origin of the unit cell

    # find the smaller and large projection of unit vectors on the axis
    aa = (abs(a1[axis]), abs(a2[axis]))
    a_large = max(aa)
    a_small = min(aa)

    assert(n_mid > 0)
    assert(d_min >= EPS)

    # points in a unit cell
    unitcell = np.linspace(rO, a_large, n_mid + 2)
    d0 = unitcell[1] - unitcell[0]
    assert(d0 >= d_min)

    # insert additional points, if needed
    pts_add = [a_small] + list(v[axis] for v in basis)
    unitcell = compose([unitcell, np.asarray(pts_add)], d_min)

    return unitcell


def lattice_mesh1d(unitcell:'unit-cell mesh', L:int):
    """ Produces a 1d mesh for 2d lattice on a single axis
        by repeating the given unit-cell mesh
    """

    a0 = unitcell[-1]
    N_u = unitcell.size
    mesh = np.empty((N_u - 1) * L + 1, dtype=float)
    mesh[:N_u] = unitcell

    n0 = N_u - 1
    for ll in range(1, L):
        iB = ll * n0 + 1
        iE = iB + n0
        mesh[iB:iE] = (unitcell + ll * a0)[1:]

    return mesh


def lattice_points(a1:'1d-array', a2:'1d-array', L1:int, L2:int, origin=(0., 0.)):
    """ Produces lattice points given two basis vectors a1 and a2,
    and number of unit-cells on each direction.
    """

    lpts = np.asarray(tuple(n1 * a1 + n2 * a2
                            for n2 in range(L2 + 1)
                            for n1 in range(L1 + 1)), dtype=float)

    return lpts + np.asarray(origin)


def mesh_points(mesh_x, mesh_y):
    """ Produce the 2d mesh points given meshes on the x and y axis.
    x is on the columns and y is on the rows.
    """
    return np.asarray(tuple((x_i, y_j)
                            for y_j in mesh_y    # rows
                            for x_i in mesh_x),  # columns
                      dtype=float)


def h_values(mesh1d:'1d-array') -> '1d-array':
    """
    Calculates the h-values:
      h_i = x_{i+1} - x_i

    with periodic boundary conditions:

      h_{-1} = x_0 - x_{-1} := h_0
      h_{N-1} = x_{N} - x_{N-1} := h_0
    """

    hh = np.empty(mesh1d.size + 1)
    hh[1:-1] = np.diff(mesh1d)
    hh[0] = hh[-1] = hh[1]
    return hh

#==============================================================================80

def plot_2meshes(xy_fine, xy_coarse, xy_mesh=None):
    x_fine, y_fine = xy_fine
    x_coarse, y_coarse = xy_coarse

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=150)

    if xy_mesh is not None:
        x_mesh, y_mesh = xy_mesh
        X, Y = np.meshgrid(x_mesh, y_mesh)
        ax.scatter(X, Y, marker='o', s=10, alpha=0.5, linewidth=0, color='black')

    X_f, Y_f = np.meshgrid(x_fine, y_fine)
    ax.scatter(X_f, Y_f, marker='+', s=15, linewidth=0.5, color='red')

    X_c, Y_c = np.meshgrid(x_coarse, y_coarse)
    ax.scatter(X_c, Y_c, marker='+', s=20, linewidth=1, color='blue')

    ax.set_aspect('equal')
    return



def axis_points(lim:Limits, dx:float, dy:float):
    xs = np.arange(*lim.x, dx, dtype=float)
    ys = np.arange(*lim.y, dy, dtype=float)
    return xs, ys


def test_mesh2d():
    limits = Limits((0, 1), (0, 1))
    lim_coarse = Limits((0.4, 0.8), (0.5, 0.8))
    lim_fine = limits
    distance_min = 1e-4

    d_coarse = (0.05, 0.05)
    d_min_coarse = min(*d_coarse)
    x_coarse, y_coarse = axis_points(lim_coarse, *d_coarse)
    N_coarse = (x_coarse.size, y_coarse.size)
    assert(N_coarse[0] > 1 and N_coarse[1] > 1)

    d_fine = (0.03, 0.03)
    d_min_fine = min(*d_fine)
    x_fine, y_fine = axis_points(lim_fine, *d_fine)

    N_fine = (x_fine.size, y_fine.size)
    assert(N_fine[0] > 1 and N_fine[1] > 1)

    d_min = min(d_min_fine, d_min_coarse)
    assert(d_min >= distance_min)


    print(f"* Fine grid: X({N_fine[0]}):{lim_fine.x}, Y({N_fine[1]}):{lim_fine.y}, "
          f"d_min = {d_min_fine:.8f}")
    print(f"* Coarse grid: X({N_coarse[0]}):{lim_coarse.x}, Y({N_coarse[1]}):{lim_coarse.y}, "
          f" d_min = {d_min_coarse:.8f}")

    x_mesh = compose([x_fine, x_coarse], distance_min)
    hx = h_values(x_mesh)
    d_min_x_mesh = hx.min()

    y_mesh = compose([y_fine, y_coarse], distance_min)
    hy = h_values(y_mesh)
    d_min_y_mesh = hy.min()

    print(f"X-mesh ({x_mesh.size}): [{x_mesh[0]}, {x_mesh[-1]}], d_min = {d_min_x_mesh:.8f}")
    print(f"Y-mesh ({y_mesh.size}): [{y_mesh[0]}, {y_mesh[-1]}], d_min = {d_min_y_mesh:.8f}")

    plot_2meshes((x_fine, y_fine), (x_coarse, y_coarse), (x_mesh, y_mesh))
    plt.show()


def magnitude(vec:'1d-array'):
    return np.sqrt(np.sum(vec**2))


def lattice_mesh2d(a1:'1d-array', a2:'1d-array', Lx:int, Ly:int,
                   xO:float = 0, yO:float = 0, n_mid:int = 1):
    """
    n_mid is the number of points between two consecutive lattice sites
    on the same lattice basis vector
    """

    assert(magnitude(a1) > EPS and magnitude(a2) > EPS)
    assert(Lx > 1 and Ly > 1)
    assert(n_mid > 0)

    n:int = n_mid + 1

    def _mesh1d(rO, L, a, n):
        n_pts = n * (L - 1) + 1
        rs = np.linspace(rO, rO + (L - 1) * a, n_pts)
        d0 = rs[1] - rs[0]
        return rs, d0

    def _mesh(rO, L, v1:float, v2:float, n):
        a1 = abs(v1)
        a2 = abs(v2)

        if a1 < EPS:
            a_min = a_max = a2
        elif a2 < EPS:
            a_min = a_max = a1
        else:
            a_ = tuple((a1, a2))
            a_min, a_max = min(a_), max(a_)

        # fine x-mesh
        rf, df = _mesh1d(rO, L, a_min, n)
        # coarse x-mesh
        rc, dc = _mesh1d(rO, L, a_max, n)

        d_min = min(df, dc)
        return compose([rf, rc], d_min)

    # x-axis
    xs = _mesh(xO, Lx, a1[0], a2[0], n)

    # y-axis
    ys = _mesh(yO, Ly, a1[1], a2[1], n)

    return xs, ys


def test_lattice_mesh2d():
    # Ag(111)
    a0 = 1.0  # = 2.89 A
    th_a = 2 * np.pi / 3  # = 120 deg
    a1 = a0 * np.array([1, 0], dtype=float)
    a2 = a0 * np.array([np.cos(th_a), np.sin(th_a)], dtype=float)

    Lx_a:int = 10
    Ly_a:int = Lx_a

    xa_0, ya_0 = 0, 0
    x_Ag, y_Ag = lattice_mesh2d(a1, a2, Lx_a, Ly_a, xa_0, ya_0, n_mid = 1)

    # NTCDA/Ag(111) in the rectangular monolayer (r-ML) phase
    # two molecules within the unit cell adopt a herringbone arrangement
    phi_b = np.pi / 6  # = 30 deg
    b1 = 4 * a1
    b2 = 6 * a2 * np.cos(phi_b)

    Lx_b:int = 3
    Ly_b:int = Lx_b

    xb_0, yb_0 = 4, 4
    x_Mol, y_Mol = lattice_mesh2d(b1, b2, Lx_b, Ly_b, xb_0, yb_0, n_mid = 2)

    plot_2meshes((x_Ag, y_Ag), (x_Mol, y_Mol))
    plt.show()


if __name__ == "__main__":
    test_mesh2d()
    test_lattice_mesh2d()
