import minh
import math
import grid_tags
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

L = 125
mass = 1
R = 8    # 8 Mpc

def main():
    f, x, y, z, mvir, rvir, id, upid = get_file_data()
    host_masses, target_ids = find_hosts(mvir, upid)
    c_a_list, b_a_list, ddim_list, dbright_list = find_neighbors(f, rvir, mvir, x, y, z, target_ids)
    scatter(c_a_list, b_a_list, ddim_list, dbright_list)


def get_file_data():
    f = minh.open('hlist_1.00000.minh')
    x, y, z, mvir, rvir, id, upid = f.read(['x', 'y', 'z', 'mvir', 'rvir', 'id', 'upid'])
    return f, x, y, z, mvir, rvir, id, upid

# ddim = ndim / ndim_avg
    # ndim_avg = all objects in simulation
    # ndim = all objects in 8 Mpc sphere around host
        # hosts are of same mass range as last coding exercise???
# dbright = nbright / nbright_avg
    # m >= 5e11 solar masses/h
# find axis ratios using NEIGHBORS in 8 Mpc, not subhaloes

def find_hosts(mvir, upid):
    """
    find objects with mass of 10e13 Msun/h that are NOT subhaloes
    """
    target_ids = np.where((mvir >= 1e13) & (upid == -1))
    host_masses = mvir[target_ids]
    return host_masses, target_ids


def check_boundary(dx, dy, dz):
    """ Can only be applied to displacements, not positions.
    """
    list = [dx, dy, dz]
    for displacement in list:
        # subhalo split and on right side, host left
        too_big = np.where(displacement > (L / 2))
        # subhalo split and on left side, host right
        too_small = np.where(displacement < (-L / 2))
        displacement[too_big] -= L
        displacement[too_small] += L
    return dx, dy, dz


def make_inertia_tensor():
    """ Create an empty 3x3 matrix to store our inertia tensor
    values.
    """
    empty_inertia_tensor = np.zeros((3, 3))
    return empty_inertia_tensor


def populate_inertia_tensor(inertia_tensor, sub_dx, sub_dy, sub_dz):
    """ Moments and Products of Inertia about various axes:
        Ixx = sum[(y^2 + z^2) * mass]
        Iyy = sum[(x^2 + z^2) * mass]
        Izz = sum[(x^2 + y^2) * mass]
        Ixy = Iyx = -sum[x * y * mass]
        Iyz = Izy = -sum[y * z * mass]
        Ixz = Izx = -sum[x * z * mass]

    We use this matrix to determine the moment of inertia for an
    arbitrarily shaped object, characterizing its shape.

    Coordinates must be displacements, not positions.
    """
    inertia_tensor[0][0] = np.sum(((sub_dy**2) + (sub_dz**2)) * mass)
    inertia_tensor[0][1] = -np.sum((sub_dx * sub_dy * mass))
    inertia_tensor[0][2] = -np.sum((sub_dx * sub_dz * mass))
    inertia_tensor[1][0] = -np.sum((sub_dx * sub_dy * mass))
    inertia_tensor[1][1] = np.sum(((sub_dx ** 2) + (sub_dz ** 2)) * mass)
    inertia_tensor[1][2] = -np.sum((sub_dy * sub_dz * mass))
    inertia_tensor[2][0] = -np.sum((sub_dx * sub_dz * mass))
    inertia_tensor[2][1] = -np.sum((sub_dy * sub_dz * mass))
    inertia_tensor[2][2] = np.sum(((sub_dx ** 2) + (sub_dy ** 2)) * mass)
    return inertia_tensor


def compute_e_values(inertia_tensor):
    """ Function computes the eigenvalues and right eigenvectors
    of a the inertia tensor. It returns an array of the eigenvalues,
    and an array of unit "length" eigenvectors.

    This inertia tensor matrix transforms a rotation vector into
    an angular momentum vector. The eigenvectors of the inertia
    tensor are the axes about which we can rotate the object
    without wobbling/procession. The eigenvalues are the moment(s)
    of inertia, which when multiplied by the angular frequency,
    characterizes the angular momentum.
    """
    evalues, evectors = np.linalg.eig(inertia_tensor)
    return evalues


def convert_to_length(evalues):
    """
    This function converts the eigenvalues into physical lengths
    for our host halo axes, and thus our axis ratios.

    Ellipsoid equations:
        Ia = (1/5)* mass * (b^2 + c^2)
        Ib = (1/5)* mass * (a^2 + c^2)
        Ic = (1/5)* mass * (a^2 + b^2)

    Solve a system of equations for a,b,c and reorder
    so a is the largest axis, and so on.
    """
    Ia = evalues[0]
    Ib = evalues[1]
    Ic = evalues[2]
    c = math.sqrt((1/2) * 5 * (1/mass) * (Ib - Ic + Ia))
    b = math.sqrt((5 * (1/mass) * Ia) - c**2)
    a = math.sqrt((5 * (1/mass) * (Ic - Ia)) + c**2)
    # shortcut!
    return np.flip(np.sort([a, b, c]))


def get_axes(sub_dx, sub_dy, sub_dz):
    """
    Returns the axis lengths largest to smallest of the halo
    delineated as a,b,c.
    """
    empty_inertia_tensor = make_inertia_tensor()
    inertia_tensor = populate_inertia_tensor(empty_inertia_tensor,
                                             sub_dx, sub_dy, sub_dz)
    evalues = compute_e_values(inertia_tensor)
    a_len, b_len, c_len = convert_to_length(
        evalues)
    c_a_ratio = c_len / a_len
    b_a_ratio = b_len / a_len
    return c_a_ratio, b_a_ratio


def total_dim_bright(mvir):
    """Find all of the dim and bright objects in the simulation.
    We are using the fact that mass is proportional to luminosity,
    of the dark matter haloes and galaxies (dark matter is not
    itself luminescent.
    """
    total_dim = len(mvir)
    total_bright = len(mvir[np.where(mvir >= 5e11)])
    return total_dim, total_bright


def find_neighbors(f, rvir, mvir, x, y, z, target_ids):
    """Find dim and bright objects within 8Mpc sphere. Compute
    axis ratios for each host using objects, not limited to subhaloes.
    """
    total_dim, total_bright = total_dim_bright(mvir)
    # Convert rvir from kpc/h to Mpc/h.
    rvir /= 1e3
    # Initialize the grid by picking the number of cells (~50) works well
    cells = 50
    cell_width = f.L / cells
    grid = grid_tags.Grid(cell_width, cells)
    # Inputs the grid will use for searching.
    points = np.array([x, y, z]).T
    tags = np.arange(len(mvir))
    grid.populate_cells(points, tags)
    # Choose the halo we want to search around. Loop over hosts
    sub_mvir_list = []
    c_a_list = []
    b_a_list = []
    ddim_list = []
    dbright_list = []
    for id in target_ids[0]:
        host_point = points[id]
        # Finally, search around the host! change host_rvir to search bubble of 8Mpc
        sub_x, sub_y, sub_z, sub_idx = grid.retrieve_tagged_members(host_point, R).T
        sub_idx = np.array(sub_idx, dtype=int)  # The index is a float by default, change.
        # Now we can use sub_idx to extract whatever subhalo properties we want
        sub_mvir = mvir[sub_idx]
        #convert positions to displacements by subtracting out position of host
        sub_dx = sub_x - host_point[0]
        sub_dy = sub_y - host_point[1]
        sub_dz = sub_z - host_point[2]
        c_a_ratio, b_a_ratio = get_axes(sub_dx, sub_dy, sub_dz)
        sub_mvir_list.append(sub_mvir)
        if len(sub_idx) >= 4:
            c_a_list.append(c_a_ratio)
            b_a_list.append(b_a_ratio)
        ddim = len(sub_idx) / total_dim
        dbright = len(sub_mvir[np.where(sub_mvir >= 5e11)]) / total_bright
        ddim_list.append(ddim)
        dbright_list.append(dbright)
    return c_a_list, b_a_list, ddim_list, dbright_list


def scatter(c_a_list, b_a_list, ddim_list, dbright_list):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), dpi=110, constrained_layout=True)
    fig.suptitle('Distributions of properties of Local Volume analogues in minh simulation')
    ax[0,0].set_xlabel(r'$\Delta_{dim}$')
    ax[0,0].set_ylabel(r'$\Delta_{bright}$')
    ax[0,1].set_xlabel(r'$\Delta_{dim}$ ')
    ax[0,1].set_ylabel(r'$(b/a)_{8Mpc}$')
    ax[1,0].set_xlabel(r'$\Delta_{dim}$')
    ax[1,0].set_ylabel(r'$(c/a)_{8Mpc}$')
    ax[1,1].set_xlabel(r'$(b/a)_{8Mpc}$')
    ax[1,1].set_ylabel(r'$(c/a)_{8Mpc}$')
    ax[0,0].hist2d(ddim_list, dbright_list, bins=30, cmap='Greys')
    ax[0,1].hist2d(ddim_list, b_a_list, bins=30, cmap='Greys')
    ax[1,0].hist2d(ddim_list, c_a_list, bins=30, cmap='Greys')
    ax[1,1].hist2d(b_a_list, c_a_list, bins=30, cmap='Greys')
    comment = mpatches.Patch(edgecolor='black', facecolor='white', label=r'M$\propto$L')
    ax[0,0].legend(handles=[comment])
    plt.show()


if __name__ == '__main__':
    main()