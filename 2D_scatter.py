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
    f, x, y, z, mvir, rvir, id, upid, M_B, mpeak, vx, vy, vz = get_file_data()
    host_masses, target_ids = find_hosts(mvir, upid, M_B)
    c_a_env, b_a_env, c_a_rvir, b_a_rvir, ddim_list, ddim_rvir_ratios, \
    dbright_list, mpeak_list, rotations = find_neighbors(f, rvir, mvir, x, y, z, target_ids, M_B, mpeak, vx, vy, vz)
    #scatter(c_a_env, b_a_env, ddim_list, dbright_list)
    #trial_plots(c_a_env, c_a_Rvir, ddim_list, mpeak_list)
    #statistics_sat_split(c_a_rvir, c_a_env, b_a_env, ddim_list, dbright_list)
    #statistics_env_split(rotations, c_a_env, b_a_env, ddim_list, dbright_list, c_a_rvir, b_a_rvir)
    statistics_percentiles(rotations, c_a_env, b_a_env, ddim_list, dbright_list, c_a_rvir, b_a_rvir)

def get_file_data():
    """ This function reads in data from a high resolution simulation
    and computes the B-band magnitude (M_B) for each halo/subhalo.
    """
    f = minh.open('L63_hlist_1.00000.minh')
    x, y, z, mvir, rvir, id, upid, mpeak, vx, vy, vz = f.read(['x', 'y', 'z',
                                            'mvir', 'rvir',
                                            'id', 'upid', 'mpeak', 'vx', 'vy', 'vz'])
    # mpeak: maximum mass a halo has ever had over the lifetime of the simulation
    M_B = match_B(mpeak, sigma, mass_def="mpeak", LF="Neuzil", data_dir='.')
    return f, x, y, z, mvir, rvir, id, upid, M_B, mpeak, vx, vy, vz

# critical vs mean density: need for universe to not be curved vs. average in a given volume
# density profile for DM halo: find a place where density drops below __x critical, mean density... that's the radius
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


def convert_to_length(evalues, i):
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
    c = ((1/2) * 5 * (1/mass) * (Ib - Ic + Ia))**(0.5)   # if evalues are too small, c b a are nan
    b = math.sqrt((5 * (1/mass) * Ia) - c**2)
    a = math.sqrt((5 * (1/mass) * (Ic - Ia)) + c**2)
    # shortcut!
    return np.flip(np.sort([a, b, c]))


def get_axes(sub_dx, sub_dy, sub_dz, i):
    """ Returns the axis lengths largest to smallest of the halo
    delineated as a,b,c.
    """
    empty_inertia_tensor = make_inertia_tensor()
    inertia_tensor = populate_inertia_tensor(empty_inertia_tensor,
                                             sub_dx, sub_dy, sub_dz)
    evalues = compute_e_values(inertia_tensor)
    a_len, b_len, c_len = convert_to_length(
        evalues, i)
    c_a_ratio = c_len / a_len
    b_a_ratio = b_len / a_len
    """
    if i == 30:
        print(evalues)
        print('c/a ratio', c_a_ratio)
        print('b/a ratio', b_a_ratio)
    """
    return c_a_ratio, b_a_ratio


# figuring out the rotation of a satellite system around a host
def sub_L(sub_dx, sub_dy, sub_dz, vx, vy, vz, inertia_tensor, minor_ax):
    """ Find L using RHR of each satellite rotating around minor axis, if all are positive or
    negative they are rotating together. Take dot product of AM vector with minor axis to get direction.
    # compute angular velocities
    # angular velocity equation: \vec{\omega_i} = \vec{r_i} \cross \vec{v_i} / |r_i|^2
    # linear velocities are in catalog as vx, vy, vz in km/s  ... are these already relative to host center?
    # omega = r x v / r^2
    # L = I x omega
    # L and omega are column vectors, I is a matrix
    # where i selects the position and velocity vectors of a subhalo relative to the host center ... sub_dx, etc


    Now I have too many values … if 40% satellites have positive rotation and 60% is negative, absolute
    value subtraction of 2 and get a value of 0.2 for this host. Higher numbers = more corotation!!

    """
    omega_xarr = (sub_dx * vx) / (np.absolute(sub_dx))**1/2
    omega_yarr = (sub_dy * vy) / (np.absolute(sub_dy))**1/2
    omega_zarr = (sub_dz * vz) / (np.absolute(sub_dz))**1/2
    L_direction = []
    # loops over all of host's subhaloes
    for i in range(len(omega_xarr)):
        omega_x = omega_xarr[i]
        omega_y = omega_yarr[i]
        omega_z = omega_zarr[i]
        omega_col = np.reshape(np.array([omega_x, omega_y, omega_z]), (3,1))
        L = np.dot(inertia_tensor, omega_col)  # actual angular momentum
        L_unit = L / np.sqrt((L[0]**2) + (L[1]**2) + (L[2]**2))    # direction of angular momentum
        L_dir = np.dot(minor_ax, L_unit)
        L_direction.append(L_dir)
    L_direction = np.asarray(L_direction)
    count = 0
    for el in L_direction:
        if el >= 0:
            count += 1
    pos_ratio = count / len(L_direction)
    neg_ratio = 1 - pos_ratio
    rotation = np.abs(pos_ratio - neg_ratio)
    return rotation

def new_tensor(sub_dx, sub_dy, sub_dz, vx, vy, vz):
    # compare to other method
    # S_ii, S_ij, S_ji, S_jj, S_jk, S_kj, S_kk, S_ik, S_ki
    # individual elements of shape tensor: Sij = sum[m_k (r_k)_i (r_k)_j] / sum[m_k]
    inertia_tensor = np.zeros((3, 3))
    inertia_tensor[0][0] = np.sum(mass * sub_dx * sub_dx) / np.sum(mass)
    inertia_tensor[0][1] = np.sum(mass * sub_dx * sub_dy) / np.sum(mass)
    inertia_tensor[0][2] = np.sum(mass * sub_dx * sub_dz) / np.sum(mass)
    inertia_tensor[1][0] = np.sum(mass * sub_dy * sub_dx) / np.sum(mass)
    inertia_tensor[1][1] = np.sum(mass * sub_dy * sub_dy) / np.sum(mass)
    inertia_tensor[1][2] = np.sum(mass * sub_dy * sub_dz) / np.sum(mass)
    inertia_tensor[2][0] = np.sum(mass * sub_dz * sub_dx) / np.sum(mass)
    inertia_tensor[2][1] = np.sum(mass * sub_dz * sub_dy) / np.sum(mass)
    inertia_tensor[2][2] = np.sum(mass * sub_dz * sub_dz) / np.sum(mass)
    # for ellipsoid of uniform density, evalues = Ma^2/5, Mb^2/5, Mc^2/5
    evalues, evectors = np.linalg.eig(inertia_tensor)
    Ia = evalues[0]
    Ib = evalues[1]
    Ic = evalues[2]
    smallest_evalue = np.where(evalues == evalues.min())[0]
    minor_ax = np.reshape(evectors[smallest_evalue], (1,3))
    a = ((5 * Ia) / mass) ** (1/2)
    b = ((5 * Ib) / mass) ** (1/2)
    c = ((5 * Ic) / mass) ** (1/2)
    a_len, b_len, c_len = np.flip(np.sort([a, b, c]))
    c_a_ratio = c_len / a_len
    b_a_ratio = b_len / a_len
    rotation = sub_L(sub_dx, sub_dy, sub_dz, vx, vy, vz, inertia_tensor, minor_ax)
    return c_a_ratio, b_a_ratio, rotation


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


def unit_tests(i):
    """Create a set of points with specific axis ratio, np.random.ran, Gaussian distribution in all 3 directions (array),
    multiply one direction by 1/2 or shift by adding a number or something, know which should be which axis.

    Take favorite velocity and position vector and see what unit L vector it gets me.
    Check that unit vectors are pointing inn same direction. 2D version: plt.quiver
    """
    # numbers 0.0 - 1.0
    test_dx = 0.1*(np.random.rand(100))  # smaller the fraction, smaller the c/a ratio
    test_dy = np.random.rand(100)
    test_dz = np.random.rand(100)
    vx_test = np.random.rand(100)
    vy_test = np.random.rand(100)
    vz_test = np.random.rand(100)
    c_a_test, b_a_test, rotations_test = new_tensor(test_dx, test_dy, test_dz, vx_test, vy_test, vz_test)
    if i == 30:
        print('ca test', c_a_test) # smaller = flatter configuration
        print('ba test', b_a_test)
        print('Ldir test', rotations_test) # getting about 1 every time, means everything is pointed in same direction


def find_neighbors(f, rvir, mvir, x, y, z, target_ids, M_B, mpeak, vx, vy, vz):
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
    c_a_env = []   # axis ratios of configurations, indicates flatness of environment/collection of halo galaxies
    b_a_env = []
    ddim_list = []
    ddim_rvir_ratios = []
    dbright_list = []
    c_a_Rvir = []     # axis ratios of individual host haloes, not environmental
    b_a_Rvir = []
    mpeak_host_list = []
    rotations = []
    i = 0
    for id in target_ids[0]:
        host_point = points[id]
        host_Rvir = rvir[id]
        # Finally, search around the host! change host_rvir to search bubble of 8Mpc
        obj_x, obj_y, obj_z, object_idx = grid.retrieve_tagged_members(host_point, R).T
        object_idx = np.array(object_idx, dtype=int)  # The index is a float by default, change.
        # Convert positions to displacements by subtracting out position of host
        obj_dx = obj_x - host_point[0]
        obj_dy = obj_y - host_point[1]
        obj_dz = obj_z - host_point[2]
        # Computing axis ratios using neighbors, not necessarily subhalo
        c_a_ratio, b_a_ratio = get_axes(obj_dx, obj_dy, obj_dz, i)
        # B magnitudes of each object around the host
        M_B_obj = M_B[object_idx]
        num_bright = len(np.where(M_B_obj < -20.5)[0])
        num_dim = len(np.where((-16 > M_B_obj) & (M_B_obj > -18))[0])
        # Compute axis ratios using subhaloes in Rvir around host
        sub_x, sub_y, sub_z, sub_idx = grid.retrieve_tagged_members(host_point, host_Rvir).T
        sub_idx = np.array(sub_idx, dtype=int)
        sub_dx = sub_x - host_point[0]
        sub_dy = sub_y - host_point[1]
        sub_dz = sub_z - host_point[2]

        sub_vx = vx[sub_idx] - vx[id]
        sub_vy = vy[sub_idx] - vy[id]
        sub_vz = vz[sub_idx] - vz[id]

        #unit_tests(i)

        c_a_rvir, b_a_rvir = get_axes(sub_dx, sub_dy, sub_dz, i)
        M_B_sub = M_B[sub_idx]
        mpeak_host = mpeak[id]
        num_bright2 = len(np.where(M_B_sub < -20.5)[0])
        num_dim2 = len(np.where((-16 > M_B_sub) & (M_B_sub > -18))[0])
        # guarantees the data arrays are the same shape
        if len(sub_idx) >= 4:
            c_a_Rvir.append(c_a_rvir)
            b_a_Rvir.append(b_a_rvir)
            c_a_env.append(c_a_ratio)
            b_a_env.append(b_a_ratio)
            # calculate with number densities
            ddim_rvir = (num_dim2 / V_sphere) / avg_dim
            ddim_rvir_ratios.append(ddim_rvir)
            ddim = (num_dim / V_sphere) / avg_dim
            dbright = (num_bright / V_sphere) / avg_bright
            ddim_list.append(ddim)
            dbright_list.append(dbright)
            mpeak_host_list.append(mpeak_host)
            c_a_new, b_a_new, rotation = new_tensor(sub_dx, sub_dy, sub_dz, sub_vx, sub_vy, sub_vz)
            rotations.append(rotation)
        """
        if i == 30:
            print('c/a NEW', c_a_new)
            print('b/a NEW', b_a_new)
            print('Ldir', rotation)
        """
        i += 1
    return c_a_env, b_a_env, c_a_Rvir, b_a_Rvir, ddim_list, ddim_rvir_ratios, dbright_list, mpeak_host_list, rotations


def scatter(c_a_list, b_a_list, ddim_list, dbright_list):
    """ Recreate 2-dimensional scatter plots from Maria's paper.
    Comparing properties of host haloes in minh simulation;
    axis ratios and luminosities.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), dpi=110, constrained_layout=True)
    fig.suptitle('Distributions of properties of Local Volume analogues in minh simulation')
    ax[0,0].set_xlabel(r'$\Delta_{dim}$')
    ax[0,0].set_ylabel(r'$\Delta_{bright}$')
    ax[0,0].set_xlabel(r'$\Delta_{dim}$ ')
    ax[0,1].set_ylabel(r'$(b/a)_{8Mpc}$')
    ax[1,0].set_xlabel(r'$\Delta_{dim}$')
    ax[1,0].set_ylabel(r'$(c/a)_{8Mpc}$')
    ax[1,1].set_xlabel(r'$(b/a)_{8Mpc}$')
    ax[1,1].set_ylabel(r'$(c/a)_{8Mpc}$')
    # hist or scatter?
    ax[0,0].hist2d(ddim_list, dbright_list, bins=30, cmap='Greys')
    ax[0,1].hist2d(ddim_list, b_a_list, bins=30, cmap='Greys')
    ax[1,0].hist2d(ddim_list, c_a_list, bins=30, cmap='Greys')
    ax[1,1].hist2d(b_a_list, c_a_list, bins=30, cmap='Greys')
    comment = mpatches.Patch(edgecolor='black', facecolor='white', label=r'M$\propto$L')
    ax[0,0].legend(handles=[comment])
    plt.show()


def trial_plots(c_a_env, c_a_Rvir, ddim_list, mpeak_list):
    """Trying out different plots that compare three host halo
    properites at a time: c/a axis ratio at 8Mpc, c/a_rvir, and
    overdensity of dim objects in the 8Mpc environment of host.
    """
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(7, 7), dpi=120, constrained_layout=True)
    fig.suptitle('Distribution of Host Halo Axis Ratios in Different Environments')
    ax[0,0].set_xlabel(r'$\Delta_{dim}$')
    ax[0,0].set_ylabel(r'$(c/a)_{host}$')
    ax[0,0].scatter(ddim_list, c_a_env, marker='o', color='red', alpha=0.25, label=r'$(c/a)_{8Mpc}$')
    ax[0,0].scatter(ddim_list, c_a_Rvir, marker='o', color='green', alpha=0.25, label=r'$(c/a)_{rvir}$')
    ax[0, 0].legend(loc="lower right", fontsize=6)

    # plot the lowest and highest 25% of hosts in c_a_Rvir
    c_a_env_arr = np.asarray(c_a_env)
    c_a_Rvir_arr = np.asarray(c_a_Rvir)
    ddim_arr = np.asarray(ddim_list)
    # np.percentile ... explore this
    small_ratio_lim = c_a_Rvir_arr < np.percentile(c_a_Rvir_arr, 25)
    large_ratio_lim = c_a_Rvir_arr > np.percentile(c_a_Rvir_arr, 75)
    ax[0, 1].set_xlabel(r'$\Delta_{dim}$')
    ax[0, 1].set_ylabel(r'$(c/a)_{8Mpc}$')
    ax[0, 1].scatter(ddim_arr[large_ratio_lim], c_a_env_arr[large_ratio_lim], marker='o', color='green', alpha=0.25, label=r'largest 25% $(c/a)_{rvir}$')
    ax[0, 1].scatter(ddim_arr[small_ratio_lim], c_a_env_arr[small_ratio_lim], marker='o', color='orange', alpha=0.25, label=r'smallest 25% $(c/a)_{rvir}$')
    ax[0, 1].legend(loc="lower right", fontsize=6)

    ax[1, 0].set_xlabel(r'$(c/a)_{rvir}$')
    ax[1, 0].set_ylabel(r'$(c/a)_{8Mpc}$')
    ax[1, 0].scatter(c_a_Rvir, c_a_env, marker='o', color='black', alpha=0.25)

    ax[1, 1].set_xlabel(r'$(c/a)_{rvir}$')
    ax[1, 1].set_ylabel(r'$Mpeak_{host}$ (Msun/h)')
    ax[1, 1].scatter(c_a_Rvir, mpeak_list, marker='o', color='blue', alpha=0.25)
    plt.show()
    # last plot indicates there aren't any high mass/highly populated flat configurations in this sim
    # that indicates a plane of satellites (host haloes)?


def group(vara, varb, varc, ax, i, j, color1, color2, per1, per2, labela, labelb, labelc):
    if i & j == 0:
        small_ratio_lim = vara < np.percentile(vara, per1)
        large_ratio_lim = vara > np.percentile(vara, per2)
        ax.scatter(varb[large_ratio_lim], varc[large_ratio_lim], marker='.', color=color1,
                         label='largest ' + str(per1) + '% ' + labela)
        ax.scatter(varb[small_ratio_lim], varc[small_ratio_lim], marker='.', color=color2,
                         label='smallest ' + str(per1) + '% ' + labela)
        ax.axis('square')
        ax.set_xlabel(labelb, fontsize=8)
        ax.set_ylabel(labelc, fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="best", fontsize=5)
    # x, z, cut on y
    if i & j == 1:
        small_ratio_lim = varc < np.percentile(varc, per1)
        large_ratio_lim = varc > np.percentile(varc, per2)

        ax.scatter(varb[large_ratio_lim], vara[large_ratio_lim], marker='.', color=color1,
                         label='largest ' + str(per1) + '% ' + labelc)
        ax.scatter(varb[small_ratio_lim], vara[small_ratio_lim], marker='.', color=color2,
                         label='smallest ' + str(per1) + '% ' + labelc)
        ax.axis('square')
        ax.set_xlabel(labelb, fontsize=8)
        ax.set_ylabel(labela, fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="best", fontsize=5)
    # y, z cut on x
    if i & j == 2:
        small_ratio_lim = varb < np.percentile(varb, per1)
        large_ratio_lim = varb > np.percentile(varb, per2)
        ax.scatter(varc[large_ratio_lim], vara[large_ratio_lim], marker='.', color=color1,
                         label='largest ' + str(per1) + '% ' + labelb)
        ax.scatter(varc[small_ratio_lim], vara[small_ratio_lim], marker='.', color=color2,
                         label='smallest ' + str(per1) + '% ' + labelb)
        ax.axis('square')
        ax.set_xlabel(labelc, fontsize=8)
        ax.set_ylabel(labela, fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(loc="best", fontsize=5)


def statistics_sat_split(c_a_Rvir, c_a_env, b_a_env, ddim_list, dbright_list):
    """ Cosmetic changes to 16 plots:

            Make plots square: equal aspect ratios

        can break into multiple plots

            Change markers to opaque small dots

        Make into a 2D histogram instead of scatter, ax[1,1].hist2d(b_a_list, c_a_list, bins=30, cmap='Greys')

            MAKE UNIVERSAL!
    """
    c_a_rvir = np.asarray(c_a_Rvir)
    c_a_8 = np.asarray(c_a_env)
    b_a_8 = np.asarray(b_a_env)
    ddim = np.asarray(ddim_list)
    dbright = np.asarray(dbright_list)

    fig, ax = plt.subplots(6, 3, figsize=(13,13), dpi=90, tight_layout=False)
    color1 = 'red'
    color2 = 'green'
    for i in range(3):
        group(c_a_rvir, ddim, dbright, ax[0][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$\Delta_{bright}$')
        group(c_a_rvir, ddim, c_a_8, ax[1][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$(c/a)_{8Mpc}$')
        group(c_a_rvir, ddim, b_a_8, ax[2][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$(b/a)_{8Mpc}$')
        group(c_a_rvir, dbright, c_a_8, ax[3][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{bright}$', r'$(c/a)_{8Mpc}$')
        group(c_a_rvir, dbright, b_a_8, ax[4][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{bright}$', r'$(b/a)_{8Mpc}$')
        group(c_a_rvir, c_a_8, b_a_8, ax[5][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$(c/a)_{8Mpc}$', r'$(b/a)_{8Mpc}$')
    fig.suptitle('Distributions of Satellites and Their Environmental Properties', y=0.95)
    plt.subplots_adjust(wspace=-0.1, hspace=0.45)
    plt.show()


def statistics_env_split(rotations, c_a_env, b_a_env, ddim_list, dbright_list, ca_rvir, ba_rvir):
    """X and Y axes are satellite properties, split on environment

        Split on different percentiles... do for each set of 3 variables

        Interested in things in one corner vs another corner of plot

        Come up with number or metric representative of MW environment and find environments in simulation that could be analogs

    Clean up code.
    """
    # environment properties, add triaxiality
    c_a_8 = np.asarray(c_a_env)
    b_a_8 = np.asarray(b_a_env)
    ddim = np.asarray(ddim_list)
    dbright = np.asarray(dbright_list)
    dratio = ddim/dbright
    # satellite properties
    ca_rvir = np.asarray(ca_rvir)
    ba_rvir = np.asarray(ba_rvir)
    rotations = np.asarray(rotations)

    fig, ax = plt.subplots(2, 3, figsize=(8, 8), dpi=100, tight_layout=True)
    color1 = 'orange'
    color2 = 'blue'
    i = 0
    j = 0
    group(c_a_8, ca_rvir, rotations, ax[0][i], i, i, color1, color2, 25, 75, r'$(c/a)_{8Mpc}$', r'$(c/a)_{rvir}$', 'rotation')
    group(b_a_8, ba_rvir, rotations, ax[0][i+1], i, i, color1, color2, 25, 75, r'$(b/a)_{8Mpc}$', r'$(b/a)_{rvir}$', 'rotation')
    group(ddim, ca_rvir, rotations, ax[0][i+2], i, i, color1, color2, 25, 75, r'$\Delta_{dim}$', r'$(c/a)_{rvir}$', 'rotation')
    group(dbright, ca_rvir, rotations, ax[1][i], i, i, color1, color2, 25, 75, r'$\Delta_{bright}$', r'$(c/a)_{rvir}$', 'rotation')
    group(dratio, ca_rvir, rotations, ax[1][i+1], i, i, color1, color2, 25, 75, r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', 'rotation')
    group(dratio, ca_rvir, ba_rvir, ax[1][i+2], i, i, color1, color2, 25, 75, r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', r'$(b/a)_{rvir}$')
    fig.suptitle('Distributions of Local Satellite Properties split on Environmental Property Percentiles',  y=0.95)
    plt.show()

def statistics_percentiles(rotations, c_a_env, b_a_env, ddim_list, dbright_list, ca_rvir, ba_rvir):
    """
    """
    # environment properties, add triaxiality
    c_a_8 = np.asarray(c_a_env)
    b_a_8 = np.asarray(b_a_env)
    ddim = np.asarray(ddim_list)
    dbright = np.asarray(dbright_list)
    dratio = ddim/dbright
    # satellite properties
    ca_rvir = np.asarray(ca_rvir)
    ba_rvir = np.asarray(ba_rvir)
    rotations = np.asarray(rotations)

    fig, ax = plt.subplots(3, 6, figsize=(14,14), dpi=100, tight_layout=False)
    color1 = 'purple'
    color2 = 'pink'
    i = 0
    j = 0
    per_list = [5, 10, 25]
    for per in per_list:
        group(c_a_8, ca_rvir, rotations, ax[i][0], j, j, color1, color2, per, 100-per, r'$(c/a)_{8Mpc}$', r'$(c/a)_{rvir}$', 'rotation')
        group(b_a_8, ba_rvir, rotations, ax[i][1], j, j, color1, color2, per, 100-per, r'$(b/a)_{8Mpc}$', r'$(b/a)_{rvir}$', 'rotation')
        group(ddim, ca_rvir, rotations, ax[i][2], j, j, color1, color2, per, 100-per, r'$\Delta_{dim}$', r'$(c/a)_{rvir}$', 'rotation')
        group(dbright, ca_rvir, rotations, ax[i][3], j, j, color1, color2, per, 100-per, r'$\Delta_{bright}$', r'$(c/a)_{rvir}$', 'rotation')
        group(dratio, ca_rvir, rotations, ax[i][4], j, j, color1, color2, per, 100-per, r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', 'rotation')
        group(dratio, ca_rvir, ba_rvir, ax[i][5], j, j, color1, color2, per, 100-per, r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', r'$(b/a)_{rvir}$')
        i+=1
    fig.suptitle('Distributions of Local Satellite Properties split on Various Environmental Property Percentiles',  y=0.95)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()





if __name__ == '__main__':
    main()