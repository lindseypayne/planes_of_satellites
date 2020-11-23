import minh
import math
import grid_tags
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from abundance_match import match_B

L = 62.5    # Mpc/h
mass = 1
R = 8*0.7   # 8 Mpc/h
V_sim = L**3
V_sphere = (4*np.pi*(R**3)) / 3
sigma = 0   # the scatter between halo mass and galaxy luminosity

def main():
    f, x, y, z, mvir, rvir, id, upid, M_B = get_file_data()
    host_masses, target_ids = find_hosts(mvir, upid, M_B)
    c_a_list, b_a_list, ddim_list, dbright_list, ddim_for_ratios, \
    c_a_Rvir, b_a_Rvir = find_neighbors(f, rvir, mvir, x, y, z, target_ids, M_B)
    scatter(c_a_list, b_a_list, ddim_list, dbright_list, ddim_for_ratios)
    plot(c_a_list, c_a_Rvir, ddim_for_ratios)


def get_file_data():
    """ This function reads in data from a high resolution simulation
    and computes the B-band magnitude (M_B) for each halo/subhalo.
    """
    f = minh.open('L63_hlist_1.00000.minh')
    x, y, z, mvir, rvir, id, upid, mpeak = f.read(['x', 'y', 'z',
                                            'mvir', 'rvir',
                                            'id', 'upid', 'mpeak'])
    # mpeak: maximum mass a halo has ever had over the lifetime of the simulation
    M_B = match_B(mpeak, sigma, mass_def="mpeak", LF="Neuzil", data_dir='.')
    return f, x, y, z, mvir, rvir, id, upid, M_B


# change to search via MAGNITUDES
def find_hosts(mvir, upid, M_B):
    """Find objects with mass of 10e13 Msun/h that are NOT subhaloes.

    Now, Hosts should be in the magnitude range -21.25 < M_B < -20.5.
    """
    target_ids = np.where((-21.25 < M_B) & (M_B < -20.5) & (upid == -1))
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
    """ Returns the axis lengths largest to smallest of the halo
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


def dim_bright_avgs(M_B):
    """ Calculate the average number density of bright objects and
    dim objects in the simulation.

    We used the fact that mass is proportional to luminosity,
    of the dark matter haloes and galaxies (dark matter is not
    itself luminescent.

    Now, find haloes from their B-band magnitudes, not masses.
    """
    avg_bright = np.sum(M_B < -20.5) / V_sim
    print('avg_bright ', avg_bright)
    avg_dim = np.sum((-16 > M_B) & (M_B > -18)) / V_sim
    print('avg_dim ', avg_dim)
    return avg_dim, avg_bright


def find_neighbors(f, rvir, mvir, x, y, z, target_ids, M_B):
    """ Find dim and bright objects within 8Mpc sphere. Compute
    axis ratios for each host using objects, not limited to subhaloes.

    dbright calculated using only haloes with M_B < -20.5
    ddim calculated from dim galaxies with -16 > M_B > -18.
    """
    avg_dim, avg_bright = dim_bright_avgs(M_B)
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
    obj_mvir_list = []
    c_a_list = []
    b_a_list = []
    ddim_list = []
    ddim_for_ratios = []
    dbright_list = []
    c_a_Rvir = []
    b_a_Rvir = []
    for id in target_ids[0]:
        host_point = points[id]
        host_Rvir = rvir[id]
        # Finally, search around the host! change host_rvir to search bubble of 8Mpc
        obj_x, obj_y, obj_z, object_idx = grid.retrieve_tagged_members(host_point, R).T
        object_idx = np.array(object_idx, dtype=int)  # The index is a float by default, change.
        # Now we can use sub_idx to extract whatever subhalo properties we want
        obj_mvir = mvir[object_idx]
        # Convert positions to displacements by subtracting out position of host
        obj_dx = obj_x - host_point[0]
        obj_dy = obj_y - host_point[1]
        obj_dz = obj_z - host_point[2]
        # Computing axis ratios using neighbors, not necessarily subhalo
        c_a_ratio, b_a_ratio = get_axes(obj_dx, obj_dy, obj_dz)
        obj_mvir_list.append(obj_mvir)
        M_B_sat = M_B[object_idx]
        num_bright = len(np.where(M_B_sat < -20.5)[0])
        num_dim = len(np.where((-16 > M_B_sat) & (M_B_sat > -18))[0])
        if len(object_idx) >= 4:
            c_a_list.append(c_a_ratio)
            b_a_list.append(b_a_ratio)
            ddim2 = (num_dim / V_sphere) / avg_dim
            ddim_for_ratios.append(ddim2)
        # Compute axis ratios using subhaloes in Rvir
        sub_x, sub_y, sub_z, sub_idx = grid.retrieve_tagged_members(host_point, host_Rvir).T
        sub_idx = np.array(sub_idx, dtype=int)
        sub_dx = sub_x - host_point[0]
        sub_dy = sub_y - host_point[1]
        sub_dz = sub_z - host_point[2]
        c_a_rvir, b_a_rvir = get_axes(sub_dx, sub_dy, sub_dz)
        if len(sub_idx) >= 4:
            c_a_Rvir.append(c_a_rvir)
            b_a_Rvir.append(b_a_rvir)
        # calculate with number densities
        ddim = (num_dim / V_sphere) / avg_dim
        dbright = (num_bright / V_sphere) / avg_bright
        ddim_list.append(ddim)
        dbright_list.append(dbright)
    # the same for this batch of hosts, meaning none have less than 4 subs
    return c_a_list, b_a_list, ddim_list, dbright_list, ddim_for_ratios, c_a_Rvir, b_a_Rvir


def scatter(c_a_list, b_a_list, ddim_list, dbright_list, ddim_for_ratios):
    """ Recreate 2-dimensional scatter plots from Maria's paper.
    Comparing properties of host haloes in minh simulation;
    axis ratios and luminosities.
    """
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
    # hist or scatter?
    ax[0,0].hist2d(ddim_list, dbright_list, bins=30, cmap='Greys')
    ax[0,1].hist2d(ddim_for_ratios, b_a_list, bins=30, cmap='Greys')
    ax[1,0].hist2d(ddim_for_ratios, c_a_list, bins=30, cmap='Greys')
    ax[1,1].hist2d(b_a_list, c_a_list, bins=30, cmap='Greys')
    comment = mpatches.Patch(edgecolor='black', facecolor='white', label=r'M$\propto$L')
    ax[0,0].legend(handles=[comment])
    plt.show()


def plot(c_a_list, c_a_Rvir, ddim_for_ratios):
    print(len(c_a_list))
    print(len(c_a_Rvir))
    print(len(ddim_for_ratios))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 7), dpi=100, tight_layout=True)
    fig.suptitle('Distribution of satellite axis ratios in Different Environments')
    ax[0,0].set_xlabel(r'$\Delta_{dim}_{8Mpc}$')
    ax[0,0].set_ylabel(r'$(c/a)_{8Mpc}$')
    ax[0,0].hist2d(ddim_for_ratios, c_a_list, bins=30, cmap='Greys')
    # ddim of host environment on x
    # c/a of host on y
    # plot points c/a of sats
    # make smallest 25% c/a sats red, rest blue

    # How do I make arrays the same shape/length for ca_sats and ddim/ca_env??
    # Confused how to input data into plotting functions
    plt.show()

if __name__ == '__main__':
    main()