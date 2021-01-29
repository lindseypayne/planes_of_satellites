import minh
import math
import grid_tags
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from abundance_match import match_B
import matplotlib.lines as mlines
from scipy import stats
import numpy.linalg as linalg

L = 62.5    # Mpc/h
mass = 1
R = 8*0.7   # 8 Mpc/h
V_sim = L**3
V_sphere = (4*np.pi*(R**3)) / 3
sigma = 0   # the scatter between halo mass and galaxy luminosity

def main():
    f, x, y, z, mvir, rvir, id, upid, M_B, mpeak, vx, vy, vz = get_file_data()
    host_masses, target_ids = find_hosts(mvir, upid, M_B)
    c_a_env, b_a_env, c_a_rvir, b_a_rvir, c_a_med, b_a_med, ddim_list, \
    dbright_list, mpeak_list, corotations = find_neighbors(f, rvir, mvir, x, y, z,
                                            target_ids, M_B, mpeak, vx, vy, vz)
    ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, ca_med, ba_med,\
    corotations = convert(c_a_env, b_a_env, c_a_rvir, b_a_rvir, c_a_med, b_a_med,
                          ddim_list, dbright_list, corotations)
    #various_plots(ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, corotations, mpeak_list)
    # user input list of arguments
    props, MW_data, labels = user_choice(ca_8, ba_8, ca_med, ba_med, ddim, dbright, dratio,
                                         ca_rvir, ba_rvir, corotations)
    sat_MW_comparison(props, MW_data, labels, ca_rvir, corotations)  # use list as arguments
    stat_test(corotations, ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir)


def various_plots(ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, corotations, mpeak_list):
    scatter(ca_8, ba_8, ddim, dbright)
    trial_plots(ca_8, ca_rvir, ddim, mpeak_list)
    statistics_sat_split(ca_rvir, ca_8, ba_8, ddim, dbright)
    statistics_env_split(corotations, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir)
    statistics_percentiles(corotations, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir)


def convert(c_a_env, b_a_env, c_a_rvir, b_a_rvir, c_a_med, b_a_med, ddim_list, dbright_list, corotations):
    # environment properties
    ca_8 = np.asarray(c_a_env)
    ba_8 = np.asarray(b_a_env)
    ca_med = np.asarray(c_a_med)
    ba_med = np.asarray(b_a_med)
    ddim = np.asarray(ddim_list)
    dbright = np.asarray(dbright_list)
    dratio = ddim / dbright
    # satellite properties
    ca_rvir = np.asarray(c_a_rvir)
    ba_rvir = np.asarray(b_a_rvir)
    corotations = np.asarray(corotations)
    return ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, ca_med, ba_med, corotations


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
def find_hosts(mvir, upid, M_B):
    """Find objects with mass of 10e13 Msun/h that are NOT subhaloes.
    Now, Hosts should be in the magnitude range -21.25 < M_B < -20.5.
    Search via magnitudes.
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
    """ Returns the c/a and b/a axis ratios for a host halo.
    Computes them using inertia tensor method AND median axes
    method.
    """
    empty_inertia_tensor = make_inertia_tensor()
    inertia_tensor = populate_inertia_tensor(empty_inertia_tensor,
                                             sub_dx, sub_dy, sub_dz)
    evalues = compute_e_values(inertia_tensor)
    a_len, b_len, c_len = convert_to_length(
        evalues, i)
    a_med, b_med, c_med = median_axes((sub_dx, sub_dy, sub_dz), w=None)
    c_a_ratio = c_len / a_len
    b_a_ratio = b_len / a_len
    c_a_med = c_med / a_med
    b_a_med = b_med/a_med
    """
    if i > 300:
        print('difference between median and tensor ca, ba ratios:', np.abs(c_a_med - c_a_ratio), np.abs(b_a_med - b_a_ratio))
        print('')
    """
    return c_a_ratio, b_a_ratio, c_a_med, b_a_med


def sub_L(sub_dx, sub_dy, sub_dz, vx, vy, vz, inertia_tensor, minor_ax):
    """ Finding the angular momentum of each satellite rotating around a host's minor axis.
    If all are sattelite L values are positive or all negative they are rotating together.
    This function then computes a statistic for how much corotation each host halo has.
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
        L_dir = np.dot(minor_ax, L_unit)   # dot product of AM vector with minor axis gets direction.
        L_direction.append(L_dir)
    L_direction = np.asarray(L_direction)
    count = 0
    for el in L_direction:
        if el >= 0:
            count += 1
    pos_ratio = count / len(L_direction)
    neg_ratio = 1 - pos_ratio
    # ??????????
    corotation = np.abs(pos_ratio - neg_ratio)
    return corotation


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
    corotation = sub_L(sub_dx, sub_dy, sub_dz, vx, vy, vz, inertia_tensor, minor_ax)
    return c_a_ratio, b_a_ratio, corotation


def dim_bright_avgs(M_B):
    """ This function calculates the average number density of bright objects and
    dim objects in the simulation. We used the fact that mass is proportional to
    luminosity for the halos and galaxies (dark matter is not itself luminescent.
    """
    # search using B band magnitudes
    avg_bright = np.sum(M_B < -20.5) / V_sim
    print('avg_bright ', avg_bright)
    avg_dim = np.sum((-16 > M_B) & (M_B > -18)) / V_sim
    print('avg_dim ', avg_dim)
    return avg_dim, avg_bright


def unit_tests(i):
    """ Creating a set of points with a specific axis ratio using np.random.ran,
    to produce a Gaussian distribution in all 3 directions (array).

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
    c_a_median = []
    b_a_median = []
    mpeak_host_list = []
    corotations = []
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
        c_a_ratio, b_a_ratio, c_a_med, b_a_med = get_axes(obj_dx, obj_dy, obj_dz, i)
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
        c_a_rvir, b_a_rvir, _, _ = get_axes(sub_dx, sub_dy, sub_dz, i)
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
            c_a_median.append(c_a_med)
            b_a_median.append(b_a_med)
            # calculate with number densities
            ddim_rvir = (num_dim2 / V_sphere) / avg_dim
            ddim_rvir_ratios.append(ddim_rvir)
            ddim = (num_dim / V_sphere) / avg_dim
            dbright = (num_bright / V_sphere) / avg_bright
            ddim_list.append(ddim)
            dbright_list.append(dbright)
            mpeak_host_list.append(mpeak_host)
            c_a_new, b_a_new, corotation = new_tensor(sub_dx, sub_dy, sub_dz, sub_vx, sub_vy, sub_vz)
            corotations.append(corotation)  # 0 means no corotation, half in one direction half in another
        i += 1
    return c_a_env, b_a_env, c_a_Rvir, b_a_Rvir, c_a_median, b_a_median, ddim_list, \
           dbright_list, mpeak_host_list, corotations


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
    """ Trying out different plots that compare three host halo
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


def group(vara, varb, varc, ax, i, j, color1, color2, per1, per2, labela, labelb, labelc, setax, upperx, uppery):
    if i & j == 0:
        if vara is not None:
            small_ratio_lim = vara < np.percentile(vara, per1)
            large_ratio_lim = vara > np.percentile(vara, per2)
            ax.scatter(varb[large_ratio_lim], varc[large_ratio_lim], marker='.', color=color1,
                       label='largest ' + str(per1) + '% ' + labela)
            ax.scatter(varb[small_ratio_lim], varc[small_ratio_lim], marker='.', color=color2,
                       label='smallest ' + str(per1) + '% ' + labela)
        else:
            ax.scatter(varb, varc, label=None)
            ax.scatter(varb, varc, label=None)
        ax.axis('square')
        if setax is True:
            ax.set_ylim(0, uppery)
            ax.set_xlim(0, upperx)
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


def statistics_sat_split(ca_rvir, ca_8, ba_8, ddim, dbright):
    """ Displays 18 plots of the distribution of local and environmental halo properties.
    Hosts that are displayed are in the top or bottom 25th percentile of a given
    satellite property, so they are essentially "split" on this variable.
    """
    fig, ax = plt.subplots(6, 3, figsize=(13,13), dpi=90, tight_layout=False)
    color1 = 'red'
    color2 = 'green'
    upperx = 1.0
    uppery = 1.0
    for i in range(3):
        group(ca_rvir, ddim, dbright, ax[0][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$\Delta_{bright}$', False, upperx, uppery)
        group(ca_rvir, ddim, ca_8, ax[1][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$(c/a)_{8Mpc}$', False, upperx, uppery)
        group(ca_rvir, ddim, ba_8, ax[2][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$(b/a)_{8Mpc}$', False, upperx, uppery)
        group(ca_rvir, dbright, ca_8, ax[3][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{bright}$', r'$(c/a)_{8Mpc}$', False, upperx, uppery)
        group(ca_rvir, dbright, ba_8, ax[4][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{bright}$', r'$(b/a)_{8Mpc}$', False, upperx, uppery)
        group(ca_rvir, ca_8, ba_8, ax[5][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$(c/a)_{8Mpc}$', r'$(b/a)_{8Mpc}$', False, upperx, uppery)
    fig.suptitle('Distributions of Satellites and Their Environmental Properties', y=0.95)
    plt.subplots_adjust(wspace=-0.1, hspace=0.45)
    plt.show()


def statistics_env_split(corotations, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir):
    """ Displays 18 plots of the distribution of host satellite properties.
    Hosts that are displayed are in the top or bottom 25th percentile of a given
    environmental host property.
    """
    dratio = ddim / dbright
    fig, ax = plt.subplots(2, 3, figsize=(8, 8), dpi=100, tight_layout=True)
    color1 = 'orange'
    color2 = 'blue'
    i = 0
    j = 0
    upperx = 1.0
    uppery = 1.0
    group(ca_8, ca_rvir, corotations, ax[0][i], i, i, color1, color2, 25, 75,
          r'$(c/a)_{8Mpc}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    group(ba_8, ba_rvir, corotations, ax[0][i+1], i, i, color1, color2, 25, 75,
          r'$(b/a)_{8Mpc}$', r'$(b/a)_{rvir}$', 'rotation', False, upperx, uppery)
    group(ddim, ca_rvir, corotations, ax[0][i+2], i, i, color1, color2, 25, 75,
          r'$\Delta_{dim}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    group(dbright, ca_rvir, corotations, ax[1][i], i, i, color1, color2, 25, 75,
          r'$\Delta_{bright}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    group(dratio, ca_rvir, corotations, ax[1][i+1], i, i, color1, color2, 25, 75,
          r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    group(dratio, ca_rvir, ba_rvir, ax[1][i+2], i, i, color1, color2, 25, 75,
          r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', r'$(b/a)_{rvir}$', False, upperx, uppery)
    fig.suptitle('Distributions of Local Satellite Properties split on '
                 'Environmental Property Percentiles',  y=0.95)
    plt.show()


def empirical_KS(full_sample, subsample, N_loops):
    """ The null-hypothesis for the KT test is that the distributions are the same.
        Thus, the lower your p value the greater the statistical evidence you have to
        reject the null hypothesis and conclude the distributions are different.
    """
    stat_KS, pvalue = stats.kstest(subsample, full_sample)
    count = 0
    for i in range(N_loops):
        num_points = len(subsample)
        # samples won't contain the same points multiple times
        ran_sample = np.random.choice(full_sample, num_points, replace=True)
        stat_emp, pvalue_emp = stats.kstest(ran_sample, full_sample)
        if stat_emp > stat_KS:
            count += 1
    updated_pvalue = count / N_loops
    return updated_pvalue, pvalue


def empirical_AD(full_sample, subsample, N_loops):
    """An approximate significance level at which the null hypothesis for the provided
    samples can be rejected. The value is floored / capped at 0.1% / 25%.
        """
    stat_AD, critvalue, sigvalue = stats.anderson_ksamp([subsample, full_sample])
    count = 0
    for i in range(N_loops):
        num_points = len(subsample)
        # samples won't contain the same points multiple times
        ran_sample = np.random.choice(full_sample, num_points, replace=True)
        stat_emp, critvalue_emp, sigvalue_emp = stats.anderson_ksamp([ran_sample, full_sample])
        if stat_emp > stat_AD:
            count += 1
    updated_pvalue = count / N_loops
    return updated_pvalue


def individual(vara, varb, ax, per1, per2, labela, labelb, setax, upperx,
               uppery, bins, subtitle, colors, xtitle, ytitle):
    sm_ratio_lim = vara < np.percentile(vara, per1)
    lg_ratio_lim = vara > np.percentile(vara, per2)
    # vara: env, varb: sat
    # add rank order, sm_ratio_lim = euc_dist < np.percentile(euc_dist, per1), no large limit
    N_loops = 100
    lg_pvalue_KS, p = empirical_KS(varb, varb[lg_ratio_lim], N_loops)
    sm_pvalue_KS, p = empirical_KS(varb, varb[sm_ratio_lim], N_loops)
    lg_pvalue_AD = empirical_AD(varb, varb[lg_ratio_lim], N_loops)
    sm_pvalue_AD = empirical_AD(varb, varb[sm_ratio_lim], N_loops)
    # each of these curves are CDFs
    ax.hist(varb[lg_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0,upperx), color=colors[0], label='top KS pvalue ' + str(round(lg_pvalue_KS,2)))
    ax.hist(varb[lg_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0, upperx),color=colors[0], label='top AD pvalue ' + str(round(lg_pvalue_AD,2)))
    ax.hist(varb[sm_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0,upperx), color=colors[1], label='bot KS pvalue ' + str(round(sm_pvalue_KS,2)))
    ax.hist(varb[sm_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0, upperx),color=colors[1], label='bot AD pvalue ' + str(round(sm_pvalue_AD,2)))
    ax.hist(varb, bins=bins, cumulative=True, density=True, histtype='step', color=colors[2])
    ax.axis('square')
    if setax is True:
        ax.set_ylim(0, uppery)
        ax.set_xlim(0, upperx)
    if subtitle is True:
        ax.set_title('split on ' + labela, fontsize=10)
    if xtitle is True:
        ax.set_xlabel(labelb, fontsize=10)
    if ytitle is True:
        ax.set_ylabel('N_hosts', fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="best", fontsize=5)

# one of two ways to select subsamples
def statistics_percentiles(corotations, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir):
    """ Creating another 18 plots that splits only on environmental variable and plots satellite
    variable properties on the x and y axes. First row of plots splits on the top and bottom 5th
    percentile, then 10th percentile, then 25th.

    First plot shows scatter distribution. Second shows histogram the statistical probability
    (at different percentage splits on the population) of a sample population belonging to
    the same distribution as the parent population. Null hypothesis is that they do.
    """
    dratio = ddim / dbright
    fig, ax = plt.subplots(3, 6, figsize=(14,14), dpi=100, tight_layout=True)
    color1 = 'purple'
    color2 = 'pink'
    i = 0
    j = 0
    upperx = 1.1
    uppery = 1.1
    per_list = [5, 10, 25]
    for per in per_list:
        group(ca_8, ca_rvir, corotations, ax[i][0], j, j, color1, color2, per, 100-per,
              r'$(c/a)_{8Mpc}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        group(ba_8, ba_rvir, corotations, ax[i][1], j, j, color1, color2, per, 100-per,
              r'$(b/a)_{8Mpc}$', r'$(b/a)_{rvir}$', 'corotation', True, upperx, uppery)
        group(ddim, ca_rvir, corotations, ax[i][2], j, j, color1, color2, per, 100-per,
              r'$\Delta_{dim}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        if i == 0 or i == 1:
            group(None, None, None, ax[i][3], j, j, color1, color2, None, None,
                  None, None, None, True, upperx, uppery)
        if i == 2:
            group(dbright, ca_rvir, corotations, ax[i][3], j, j, color1, color2, per, 100-per,
                  r'$\Delta_{bright}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        group(dratio, ca_rvir, corotations, ax[i][4], j, j, color1, color2, per, 100-per,
              r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        group(dratio, ca_rvir, ba_rvir, ax[i][5], j, j, color1, color2, per, 100-per,
              r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', r'$(b/a)_{rvir}$', True, upperx, uppery)
        i+=1
    fig.suptitle('Scatter Distributions of Satellite Properties split on Environmental '
                 'Property Percentiles',  y=0.95)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()

    #### above: scatter, below: histogram ####

    fig, ax = plt.subplots(3, 6, figsize=(14, 14), dpi=100, tight_layout=True)
    i = 0
    upperx = 1.0
    uppery = 1.0
    bins = 200
    per_list = [5, 10, 25]
    colors5 = ['indianred', 'deepskyblue', 'black']
    colors10 = ['forestgreen', 'purple', 'black']
    colors25 = ['crimson', 'darkturquoise', 'black']
    colors = [colors5, colors10, colors25]
    ytitle = False
    xtitle = False
    setax = True
    for per in per_list:
        if i == 0:
            title = True
        if i == 1 or i == 2:
            title = False
        if i == 2:
            xtitle = True
        individual(ca_8, ca_rvir, ax[i][0], per, 100 - per, r'$(c/a)_{8Mpc}$',
              r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, True)
        individual(ca_8, corotations, ax[i][1], per, 100 - per, r'$(c/a)_{8Mpc}$',
              r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        individual(ddim, ca_rvir, ax[i][2], per, 100 - per, r'$\Delta_{dim}$',
              r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        individual(ddim, corotations, ax[i][3], per, 100 - per, r'$\Delta_{dim}$',
              r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        individual(dratio, ca_rvir, ax[i][4], per, 100 - per, r'$\Delta_{ratio}$',
              r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        individual(dratio, corotations, ax[i][5], per, 100 - per, r'$\Delta_{ratio}$',
              r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        i += 1
    fig.suptitle('Histogram Distributions of Satellite Properties Split on Environmental '
                 'Property Percentiles',
                 y=0.97)
    line1 = mlines.Line2D([], [], color='indianred', label='top 5%')
    line2 = mlines.Line2D([], [], color='deepskyblue', label='bottom 5%')
    line3 = mlines.Line2D([], [], color='forestgreen', label='top 10%')
    line4 = mlines.Line2D([], [], color='purple', label='bottom 10%')
    line5 = mlines.Line2D([], [], color='crimson', label='top 25%')
    line6 = mlines.Line2D([], [], color='darkturquoise', label='bottom 25%')
    line7 = mlines.Line2D([], [], color='black', label='all hosts')
    fig.legend(handles=[line1, line2, line3, line4, line5, line6, line7], fontsize=6)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()


def statistics_MW_rank(corotations, ca_rvir, euc_dist):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10), dpi=75, tight_layout=False)
    i = 0
    upperx = 1.0
    uppery = 1.0
    bins = 200
    per_list = [5, 10, 25]
    colors5 = ['indianred', 'deepskyblue', 'black']
    colors10 = ['forestgreen', 'purple', 'black']
    colors25 = ['crimson', 'darkturquoise', 'black']
    colors = [colors5, colors10, colors25]
    ytitle = False
    xtitle = False
    setax = True
    for per in per_list:
        if i == 0:
            title = True
        if i == 1:
            title = False
            xtitle = True
        individual(euc_dist, ca_rvir, ax[i][0], per, 100 - per, 'similarity to MW',
                   r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, True)
        individual(euc_dist, corotations, ax[i][1], per, 100 - per, 'similarity to MW',
                   r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        i += 1
    fig.suptitle('Histogram Distributions of Satellite Properties Split on Rank-Order Similarity to Milky Way',
                 y=0.97)
    line1 = mlines.Line2D([], [], color='indianred', label='top 5%')
    line2 = mlines.Line2D([], [], color='deepskyblue', label='bottom 5%')
    line3 = mlines.Line2D([], [], color='forestgreen', label='top 10%')
    line4 = mlines.Line2D([], [], color='purple', label='bottom 10%')
    line5 = mlines.Line2D([], [], color='crimson', label='top 25%')
    line6 = mlines.Line2D([], [], color='darkturquoise', label='bottom 25%')
    line7 = mlines.Line2D([], [], color='black', label='all hosts')
    fig.legend(handles=[line1, line2, line3, line4, line5, line6, line7], fontsize=6)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()


def stat_test(corotations, ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir):
    """Testing the validity of the upper-bounded KS test statistic by comparing its pvalue
    to the pvalue generated from a combination KS-Monte Carlo test, generating random
    points from the parent sample according to the number of points in the subsample.

    Plot also includes a side-by-side panel of the AD-Monte Carlo empirical pvalue as
    amother statistical comparison check.
    """
    fig, ax = plt.subplots(2, 2, dpi=150)
    fig.suptitle('Satellite to Environment Probability Values with Random Sampling')
    print('')
    percent = int(input("Enter a percentage between 0 and 100: "))
    print('')
    env = [ca_8, ba_8, ddim, dbright, dratio]
    env_label = [r'$(c/a)_{8Mpc}$', r'$(b/a)_{8Mpc}$', r'$\Delta_{dim}$', r'$\Delta_{bright}$', r'$\Delta_{ratio}$']
    sat = [ca_rvir, ba_rvir, corotations]
    sat_label = [r'$(c/a)_{rvir}$', r'$(b/a)_{rvir}$' , r'$f_{corotation}$']
    a = int(input("Choose an environmental property, ca_8, ba_8, ddim, dbright, dratio by choosing 0,1,2,3, or 4: "))
    print('')
    b = int(input("Choose a satellite property, ca_rvir, ba_rvir, corotations by entering 0,1, or 2: "))
    vara = env[a]
    varb = sat[b]
    labela = env_label[a]
    labelb = sat_label[b]
    N_loops = [1, 10, 100, 10**3, 10**4, 3*(10**4)]
    bot_ratio_lim = vara < np.percentile(vara, percent)
    top_ratio_lim = vara > np.percentile(vara, 100-percent)
    top_KS_pvalues = []
    bot_KS_pvalues = []
    top_AD_pvalues = []
    bot_AD_pvalues = []
    top_og_KS = []
    bot_og_KS = []
    for N in N_loops:
        top_KS, og1 = empirical_KS(varb, varb[top_ratio_lim], N)
        bot_KS, og2 = empirical_KS(varb, varb[bot_ratio_lim], N)
        top_AD = empirical_AD(varb, varb[top_ratio_lim], N)
        bot_AD = empirical_AD(varb, varb[bot_ratio_lim], N)
        top_KS_pvalues.append(top_KS)
        bot_KS_pvalues.append(bot_KS)
        top_AD_pvalues.append(top_AD)
        bot_AD_pvalues.append(bot_AD)
        top_og_KS.append(og1)
        bot_og_KS.append(og2)
    ax[0][0].set_title('KS Test Satellite ' + labelb + ' Split on ' + labela, fontsize=7)
    ax[0][0].plot(N_loops, top_KS_pvalues, marker='.', color='red', label='top '+str(percent)+'%')
    ax[0][0].plot(N_loops, top_og_KS, marker='.', color='blue', label='no randomization')
    ax[0][0].set_ylabel('pvalue')
    ax[1][0].plot(N_loops, bot_KS_pvalues, marker='.', color='orange', label='bottom '+str(percent)+'%')
    ax[1][0].plot(N_loops, bot_og_KS, marker='.', color='blue', label='no randomization')
    ax[1][0].set_xlabel('N_loops')
    ax[1][0].set_ylabel('pvalue')
    ax[0][1].set_title('AD Test Satellite ' + labelb + ' Split on ' + labela, fontsize=7)
    ax[0][1].plot(N_loops, top_AD_pvalues, marker='.', color='red', label='top '+str(percent)+'%')
    ax[1][1].plot(N_loops, bot_AD_pvalues, marker='.', color='orange', label='bottom '+str(percent)+'%')
    ax[1][1].set_xlabel('N_loops')
    j = 0
    k = 0
    for i in range(4):
        if i == 1 or i == 3:
            k = 1
        if i == 2:
            j = 1
            k = 0
        ax[j][k].set_ylim(10**-3, 10**1)
        ax[j][k].legend(loc="best", fontsize=5)
        ax[j][k].set_xscale("log")
        ax[j][k].set_yscale("log")
    plt.show()


def project(x, y, z, px_hat, py_hat, pz_hat):
    """ project projects the vectors given by (x, y, z) along the orthognoal
    unit vectors px_hat, py_hat, and pz_hat.
    """
    vec = np.array([x, y, z]).T
    return np.dot(vec, px_hat), np.dot(vec, py_hat), np.dot(vec, pz_hat)


def ellipse_axes(points, w=None):
    x, y, z = points
    if w is None: w = np.ones(len(x))

    M11, M12, M13 = np.sum(x*x*w), np.sum(x*y*w), np.sum(x*z*w)
    M21, M22, M23 = M12,           np.sum(y*y*w), np.sum(y*z*w)
    M31, M32, M33 = M13,           M23,           np.sum(z*z*w)
    M = np.array([
        [M11,M12,M13],
        [M21,M22,M23],
        [M31,M32,M33]
    ])
    lamb, vec = linalg.eig(M)
    axes = np.sqrt(5 * lamb / np.sum(w))
    return axes, vec, lamb


def median_axes(points, w=None):
    """ median_axes takes a set of points (x, y, z) and weights, w,
    and returns the median length along the three axes of the point distribution
    and the vectors pointing along each of these axes
    """
    axes, vec, evalues = ellipse_axes(points, w)
    x, y, z = points
    x0, x1, x2 = project(x, y, z, vec[0], vec[1], vec[2])
    def med(xi): return np.median(np.abs(xi))
    # smallest to largest, then reversed
    return np.flip(np.sort([med(x0), med(x1), med(x2)]))


# change ca_8 to ca_med_8!!!!
def user_choice(ca_8, ba_8, ca_med, ba_med, ddim, dbright, dratio, ca_rvir, ba_rvir, corotations):
    """Asks user for inputted host properties to make a list of variables
    for the argument of sat_MW_comparison.
    """
    props = [ca_8, ba_8, ca_med, ba_med, ddim, dbright, dratio, ca_rvir, ba_rvir, corotations]
    # from table 3 Neuzil et al 2019
    MW = [0.163, 0.786, 0.163, 0.786, 1.699, 5.190, 1.699/5.190, 1, 1, 1]
    labels = [r'$(c/a)_{8Mpc}$', r'$(b/a)_{8Mpc}$', r'$(c/a)_{median}$', r'$(b/a)_{median}$',
              r'$\Delta_{dim}$', r'$\Delta_{bright}$', r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$',
              r'$(b/a)_{rvir}$', r'$f_{corotation}$']
    PROPS = [ca_med, ba_med, ddim, dratio]
    MW = [ 0.163, 0.786, 1.699, 1.699/5.190]
    labels = []
    return PROPS, MW, labels
    propsa = []
    MWa = []
    labelsa = []
    for i in range(len(props)):
        a = str(input("Choose properties by entering a number 0-9 corresponding to ca_8, ba_8, "
                      "ca_median, ba_median, ddim, dbright, dratio, ca_rvir, ba_rvir, or corotations. "
                      "Enter DONE after final choice or ALL if you'd like all data considered: "))
        print('')
        if i == 0:
            while a == 'DONE':
                a = str(input('User must enter ALL or a number 0-9: '))
                print('')
            if a == 'ALL':
                return props, MW, labels
        if i != 0:
            if a == 'DONE':
                break
            while a == 'ALL':
                a = str(input('User must enter DONE or a number 0-9: '))
                print('')
        if a != 'DONE' or 'ALL':
            a = int(a)
            propsa.append(props[a])
            MWa.append(MW[a])
            labelsa.append(labels[a])
    #return propsa, MWa, labelsa


# two of two ways to select subsamples
def sat_MW_comparison(props, MW_data, labels, ca_rvir, corotations):
    """This function ranks the catalog's host halos
    in similarity to the Milky Wway.
    """
    # normalize arrays between 0 and 1
    dist_arrays = []
    i = 0
    for prop in props:
        prop = np.asarray(prop)
        MW_data = np.asarray(MW_data)
        std = np.std(prop)  # a measure of the spread of a distribution
        std2 = np.std(MW_data)
        prop /= std
        MW_data /= std2
        norm_prop = (prop - np.amin(prop)) / (np.amax(prop) - np.amin(prop))
        norm_MW = (MW_data - np.amin(MW_data)) / (np.amax(MW_data) - np.amin(MW_data))
        dist = np.square(norm_prop - norm_MW[i])
        dist_arrays.append(dist)
        i += 1
    sum = 0
    for dist in dist_arrays:
        sum += dist
    euc_dist = np.sqrt(sum)  # using the euclidean distance equation
    most_similar_halo = np.amin(euc_dist)
    least_similar_halo = np.amax(euc_dist)
    most_index = np.argmax(euc_dist)
    least_index = np.argmin(euc_dist)
    print('')
    print(euc_dist)
    print('least distance:', most_similar_halo, 'most distance:', least_similar_halo)
    print('most similar halo index:', most_index, 'least similar halo index:', least_index)
    statistics_MW_rank(corotations, ca_rvir, euc_dist)

    # paper test!! worked

def KS_2D():
    """

    :return:
    """
    print('')

if __name__ == '__main__':
    main()