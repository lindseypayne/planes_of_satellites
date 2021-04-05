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
import KS2D
import numpy.random as random

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
    dbright_list, mpeak_list, corotations, minor_ax_vir, minor_ax_8, L, \
    sat_x, sat_y, sat_z, IDs = find_neighbors(f, rvir, mvir, x, y, z, target_ids,
                                         M_B, mpeak, vx, vy, vz)

    ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, ca_med, ba_med,\
    corotations, all_props, all_labels, minor_vir, minor_8, env_props, env_labels = convert(c_a_env, b_a_env, c_a_rvir,
    b_a_rvir, c_a_med, b_a_med, ddim_list, dbright_list, corotations, minor_ax_vir, minor_ax_8)

    #various_plots(ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, corotations, mpeak_list)
    """
    euc_dist = []
    sim_labels = []
    for i in range(4):
        props, MW_data, labels = user_choice(ca_med, ba_med, ddim, dbright, dratio)
        euc = sat_MW_comparison(props, all_props, MW_data, labels, all_labels, ca_rvir, ba_rvir, corotations)
        euc_dist.append(np.asarray(euc))
        sim_labels.append(labels)
        """
    #alignment(all_labels, all_props, [c_a_med, b_a_med], [0.163, 0.786], ca_rvir, minor_vir, minor_8, 5)
    #stat_test(corotations, ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir)
    #KS_2D(ca_rvir, corotations, [ca_8, ba_8, ddim, dratio],
          #[r'$(c/a)_{8Mpc}$', r'$(b/a)_{8Mpc}$', r'$\Delta_{dim}$', r'$\Delta_{ratio}$'], 5)
    #euc_dist = np.array([1,2])
    #labels = ''
    #sat_sep([sat_x, sat_y, sat_z], L, IDs, np.asarray(euc_dist), sim_labels, env_props, env_labels)
    #test_sep()
    test_N_sats()



def various_plots(ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, corotations, mpeak_list):
    #scatter2d(ca_8, ba_8, ddim, dbright)
    #trial_plots(ca_8, ca_rvir, ddim, mpeak_list)
    #statistics_sat_split(ca_rvir, ca_8, ba_8, ddim, dbright)
    #statistics_env_split(corotations, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir)
    statistics_percentiles(corotations, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir)


def convert(c_a_env, b_a_env, c_a_rvir, b_a_rvir, c_a_med, b_a_med, ddim_list, dbright_list, corotations, minor_ax_vir, minor_ax_8):
    # environment properties
    ca_8 = np.copy(c_a_env)
    ba_8 = np.copy(b_a_env)
    ca_med = np.copy(c_a_med)
    ba_med = np.copy(b_a_med)
    ddim = np.copy(ddim_list)
    dbright = np.copy(dbright_list)
    dratio = ddim / dbright
    # satellite properties
    ca_rvir = np.copy(c_a_rvir)
    ba_rvir = np.copy(b_a_rvir)
    corotations = np.copy(corotations)
    all_props = [ca_med, ba_med, ca_rvir, ba_rvir, ddim, dbright, dratio]
    all_labels = ['ca_med', 'ba_med', 'ca_rvir', 'ba_rvir', 'ddim', 'dbright', 'dratio']
    env_props = [ca_8, ba_8, ddim, dratio]
    env_labels = [r'$c/a_{8}$', r'$b/a_{8}$', r'$\Delta_{dim}$', r'$\Delta_{ratio}$']
    minor_vir = np.asarray(minor_ax_vir)
    minor_8 = np.asarray(minor_ax_8)
    return ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir, ca_med, ba_med, corotations, \
           all_props, all_labels, minor_vir, minor_8, env_props, env_labels


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
    L_vec= []
    # loops over all of host's subhaloes
    for i in range(len(omega_xarr)):
        omega_x = omega_xarr[i]
        omega_y = omega_yarr[i]
        omega_z = omega_zarr[i]
        omega_col = np.reshape(np.array([omega_x, omega_y, omega_z]), (3,1))
        L = np.dot(inertia_tensor, omega_col)  # actual angular momentum
        L_vec.append(np.array([L[0][0], L[1][0], L[2][0]]))
        L_unit = L / np.sqrt((L[0]**2) + (L[1]**2) + (L[2]**2))    # direction of angular momentum
        L_dir = np.dot(minor_ax, L_unit)[0]   # dot product of AM vector with minor axis gets direction.
        L_direction.append(L_dir)
    L_direction = np.asarray(L_direction)
    L_vec = np.asarray(L_vec)
    count = 0
    for el in L_direction:
        if el >= 0:
            count += 1
    pos_ratio = count / len(L_direction)
    neg_ratio = 1 - pos_ratio
    # ??????????
    corotation = np.abs(pos_ratio - neg_ratio)
    return corotation, L_vec


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
    smallest_evalue = np.where(evalues == evalues.min())[0]
    minor_ax = np.reshape(evectors[smallest_evalue], (1,3))
    corotation, L = sub_L(sub_dx, sub_dy, sub_dz, vx, vy, vz, inertia_tensor, minor_ax)
    return corotation, minor_ax, L


def dim_bright_avgs(M_B):
    """ This function calculates the average number density of bright objects and
    dim objects in the simulation. We used the fact that mass is proportional to
    luminosity for the halos and galaxies (dark matter is not itself luminescent.
    """
    # search using B band magnitudes
    avg_bright = np.sum(M_B < -20.5) / V_sim
    avg_dim = np.sum((-16 > M_B) & (M_B > -18)) / V_sim
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
    rotations_test, minor_ax, L = new_tensor(test_dx, test_dy, test_dz, vx_test, vy_test, vz_test)


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
    ca_rvir = []     # axis ratios of individual host haloes, not environmental
    ba_rvir = []
    ca_median = []
    ba_median = []
    ca_env = []  # axis ratios of configurations, indicates flatness of environment/collection of halo galaxies
    ba_env = []
    ddim_list = []
    dbright_list = []
    mpeak_host_list = []
    corotations = []
    minor_ax_vir = []
    minor_ax_8 = []
    L_vec = []
    sat_x = []
    sat_y = []
    sat_z = []
    ID = []
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
        obj_vx = vx[object_idx] - vx[id]
        obj_vy = vy[object_idx] - vy[id]
        obj_vz = vz[object_idx] - vz[id]

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
        #unit_tests(i)
        c_a_rvir, b_a_rvir, _, _ = get_axes(sub_dx, sub_dy, sub_dz, i)
        mpeak_host = mpeak[id]
        # guarantees the data arrays are the same shape
        if len(sub_idx) >= 4:
            ID.append(id)
            sat_x.append(sub_x - host_point[0])
            sat_y.append(sub_y - host_point[1])
            sat_z.append(sub_z - host_point[2])
            sub_vx = vx[sub_idx] - vx[id]
            sub_vy = vy[sub_idx] - vy[id]
            sub_vz = vz[sub_idx] - vz[id]
            """
            for k in range(len(sub_dx)):
                sat_x.append(sub_dx[k])
                sat_y.append(sub_dy[k])
                sat_z.append(sub_dz[k])"""
            ca_rvir.append(c_a_rvir)
            ba_rvir.append(b_a_rvir)
            ca_env.append(c_a_ratio)
            ba_env.append(b_a_ratio)
            ca_median.append(c_a_med)
            ba_median.append(b_a_med)
            # calculate with number densities
            ddim = (num_dim / V_sphere) / avg_dim
            dbright = (num_bright / V_sphere) / avg_bright
            ddim_list.append(ddim)
            dbright_list.append(dbright)
            mpeak_host_list.append(mpeak_host)
            corotation, minor_vir, L = new_tensor(sub_dx, sub_dy, sub_dz, sub_vx, sub_vy, sub_vz)
            _, minor_8, _ = new_tensor(obj_dx, obj_dy, obj_dz, obj_vx, obj_vy, obj_vz)
            minor_ax_vir.append(minor_vir)
            minor_ax_8.append(minor_8)
            corotations.append(corotation)  # 0 means no corotation, half in one direction half in another
            L_vec.append(L)

        i += 1
    return ca_env, ba_env, ca_rvir, ba_rvir, ca_median, ba_median, ddim_list, \
           dbright_list, mpeak_host_list, corotations, minor_ax_vir, minor_ax_8, \
           np.asarray(L_vec), np.asarray(sat_x), np.asarray(sat_y), np.asarray(sat_z), ID

"""
VERY SMALL AXIS RATIO ON PLOT
ca_rvir: 0.003143141604523448, host_idx: 298
[ 0.          0.00223017 -0.09285975  0.02067041] dx
[ 0.         -0.00110054  0.07592964  0.0803194 ] dy
[ 0.         -0.00190926  0.1187706  -0.00902939] dz
4 num_sats
4
"""

def scatter2d(c_a_list, b_a_list, ddim_list, dbright_list):
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


def scatter1d(vara, varb, varc, ax, i, j, color1, color2, per1, per2, labela, labelb, labelc, setax, upperx, uppery):
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
        scatter1d(ca_rvir, ddim, dbright, ax[0][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$\Delta_{bright}$', False, upperx, uppery)
        scatter1d(ca_rvir, ddim, ca_8, ax[1][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$(c/a)_{8Mpc}$', False, upperx, uppery)
        scatter1d(ca_rvir, ddim, ba_8, ax[2][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{dim}$', r'$(b/a)_{8Mpc}$', False, upperx, uppery)
        scatter1d(ca_rvir, dbright, ca_8, ax[3][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{bright}$', r'$(c/a)_{8Mpc}$', False, upperx, uppery)
        scatter1d(ca_rvir, dbright, ba_8, ax[4][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$\Delta_{bright}$', r'$(b/a)_{8Mpc}$', False, upperx, uppery)
        scatter1d(ca_rvir, ca_8, ba_8, ax[5][i], i, i, color1, color2, 25, 75, r'$(c/a)_{rvir}$', r'$(c/a)_{8Mpc}$', r'$(b/a)_{8Mpc}$', False, upperx, uppery)
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
    scatter1d(ca_8, ca_rvir, corotations, ax[0][i], i, i, color1, color2, 25, 75,
          r'$(c/a)_{8Mpc}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    scatter1d(ba_8, ba_rvir, corotations, ax[0][i+1], i, i, color1, color2, 25, 75,
          r'$(b/a)_{8Mpc}$', r'$(b/a)_{rvir}$', 'rotation', False, upperx, uppery)
    scatter1d(ddim, ca_rvir, corotations, ax[0][i+2], i, i, color1, color2, 25, 75,
          r'$\Delta_{dim}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    scatter1d(dbright, ca_rvir, corotations, ax[1][i], i, i, color1, color2, 25, 75,
          r'$\Delta_{bright}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    scatter1d(dratio, ca_rvir, corotations, ax[1][i+1], i, i, color1, color2, 25, 75,
          r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', 'rotation', False, upperx, uppery)
    scatter1d(dratio, ca_rvir, ba_rvir, ax[1][i+2], i, i, color1, color2, 25, 75,
          r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', r'$(b/a)_{rvir}$', False, upperx, uppery)
    fig.suptitle('Local Satellite Properties Split on '
                 'Environmental Property Percentiles',  y=0.95)
    plt.show()


def p_certainty(full_sample, subsample, N_loops, iterations):
    pvals = []
    for i in range(iterations):
        p, _ = empirical_KS(full_sample, subsample, N_loops)
        pvals.append(p)
    np.asarray(pvals)
    mean_p = np.mean(pvals)
    std_p = np.std(pvals)
    label = 'for ' + str(N_loops) + ' loop, the std is ' + str(std_p) + ' and mean pvalue is ' + str(mean_p)
    return std_p, label


def sorted_ks(x, y):
    """ _sorted_ks returns the KS statistic between x and y, assuming that x has
    been pre-sorted. It also uses a more efficient alogrithm for computing
    the test statisitc. When used as part of a Monte Carlo test, it out-performs
    scipy's implementation by a factor of four.
    """
    y = np.sort(y)
    y_idx_right = np.arange(1, len(y)+1) / len(y)
    y_idx_left = np.arange(0, len(y)) / len(y)
    x_idx_right = np.searchsorted(x, y, "right") / len(x)
    x_idx_left = np.searchsorted(x, y, "left") / len(x)
    diff = np.maximum(x_idx_left - y_idx_left, y_idx_right - x_idx_right)
    return np.max(diff)

# time python3 2D.scatter
import time
# t0 = time.time()
# t1 = time.time()
# print(t1 - t0) print("%.3f s passed" % (t1 - t0))

def empirical_KS(full_sample, subsample, N_loops):
    """ The null-hypothesis for the KT test is that the distributions are the same.
        Thus, the lower your p value the greater the statistical evidence you have to
        reject the null hypothesis and conclude the distributions are different.
    """
    stat_KS, pvalue = stats.kstest(subsample, full_sample)
    #stat_KS = sorted_ks(subsample, full_sample)
    count = 0
    # loop over ~ 20 times, more,
    # take mean pvalue and std of pvalues
    # tells us how uncertain are in p due to randomness
    # check several times
    # pass in loops, 1 is default
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


def histenv(vara, varb, ax, per1, per2, labela, labelb, setax, upperx,
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
    """ax.hist(varb[lg_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0,upperx), color=colors[0], label='KS pvalue ' + str(round(lg_pvalue_KS,2)))
    ax.hist(varb[lg_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0, upperx),color=colors[0], label='AD pvalue ' + str(round(lg_pvalue_AD,2)))"""
    ax.hist(varb[sm_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0,upperx), color=colors[1], label='KS pvalue ' + str(round(sm_pvalue_KS,2)))
    ax.hist(varb[sm_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0, upperx),color=colors[1], label='AD pvalue ' + str(round(sm_pvalue_AD,2)))
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
        scatter1d(ca_8, ca_rvir, corotations, ax[i][0], j, j, color1, color2, per, 100-per,
              r'$(c/a)_{8Mpc}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        scatter1d(ba_8, ba_rvir, corotations, ax[i][1], j, j, color1, color2, per, 100-per,
              r'$(b/a)_{8Mpc}$', r'$(b/a)_{rvir}$', 'corotation', True, upperx, uppery)
        scatter1d(ddim, ca_rvir, corotations, ax[i][2], j, j, color1, color2, per, 100-per,
              r'$\Delta_{dim}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        if i == 0 or i == 1:
            scatter1d(None, None, None, ax[i][3], j, j, color1, color2, None, None,
                  None, None, None, True, upperx, uppery)
        if i == 2:
            scatter1d(dbright, ca_rvir, corotations, ax[i][3], j, j, color1, color2, per, 100-per,
                  r'$\Delta_{bright}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        scatter1d(dratio, ca_rvir, corotations, ax[i][4], j, j, color1, color2, per, 100-per,
              r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', 'corotation', True, upperx, uppery)
        scatter1d(dratio, ca_rvir, ba_rvir, ax[i][5], j, j, color1, color2, per, 100-per,
              r'$\Delta_{ratio}$', r'$(c/a)_{rvir}$', r'$(b/a)_{rvir}$', True, upperx, uppery)
        i+=1
    fig.suptitle('Scatter of Satellite Properties Split on Environmental '
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
        histenv(ca_8, ca_rvir, ax[i][0], per, 100 - per, r'$(c/a)_{8Mpc}$',
              r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, True)
        histenv(ca_8, corotations, ax[i][1], per, 100 - per, r'$(c/a)_{8Mpc}$',
              r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(ddim, ca_rvir, ax[i][2], per, 100 - per, r'$\Delta_{dim}$',
              r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(ddim, corotations, ax[i][3], per, 100 - per, r'$\Delta_{dim}$',
              r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(dratio, ca_rvir, ax[i][4], per, 100 - per, r'$\Delta_{ratio}$',
              r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(dratio, corotations, ax[i][5], per, 100 - per, r'$\Delta_{ratio}$',
              r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        i += 1
    fig.suptitle('Histogram of Satellite Properties Split on Environmental '
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



def stat_test(corotations, ca_8, ba_8, ddim, dbright, dratio, ca_rvir, ba_rvir):
    """Testing the validity of the upper-bounded KS test statistic by comparing its pvalue
    to the pvalue generated from a combination KS-Monte Carlo test, generating random
    points from the parent sample according to the number of points in the subsample.

    Plot also includes a side-by-side panel of the AD-Monte Carlo empirical pvalue as
    amother statistical comparison check.
    """
    fig, ax = plt.subplots(2, 2, dpi=150)
    fig.suptitle('Testing Probability Values with Random Sampling for Split on Environmental Property')
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
    #N_loops = [1, 10, 100]
    N_loops = [1, 10, 100, 10**3, 10**4, 3*(10**4)]
    bot_ratio_lim = vara < np.percentile(vara, percent)   # bot = most similar
    top_ratio_lim = vara > np.percentile(vara, 100-percent)
    top_KS_p = []
    bot_KS_p = []
    top_AD_p = []
    bot_AD_p = []
    top_og_KS = []
    bot_og_KS = []
    stds = []
    for N in N_loops:
        top_KS, og1 = empirical_KS(varb, varb[top_ratio_lim], N)
        bot_KS, og2 = empirical_KS(varb, varb[bot_ratio_lim], N)
        top_AD = empirical_AD(varb, varb[top_ratio_lim], N)
        bot_AD = empirical_AD(varb, varb[bot_ratio_lim], N)
        top_KS_p.append(top_KS)
        bot_KS_p.append(bot_KS)
        top_AD_p.append(top_AD)
        bot_AD_p.append(bot_AD)
        top_og_KS.append(og1)
        bot_og_KS.append(og2)
        std_p, label = p_certainty(varb, varb[bot_ratio_lim], N, 20)
        stds.append(std_p)
        print(label)
    ax[0][0].set_title('KS Test Satellite ' + labelb + ' Split on ' + labela, fontsize=7)
    ax[0][0].plot(N_loops, top_KS_p, marker='.', color='red', label='top '+str(percent)+'%')
    ax[0][0].plot(N_loops, top_og_KS, marker='.', color='blue', label='no randomization')
    ax[0][0].set_ylabel('pvalue')
    ax[1][0].plot(N_loops, bot_KS_p, marker='.', color='orange', label='bottom '+str(percent)+'%')
    ax[1][0].plot(N_loops, bot_og_KS, marker='.', color='blue', label='no randomization')
    ax[1][0].set_xlabel('N_loops')
    ax[1][0].set_ylabel('pvalue')
    ax[0][1].set_title('AD Test Satellite ' + labelb + ' Split on ' + labela, fontsize=7)
    ax[0][1].plot(N_loops, top_AD_p, marker='.', color='red', label='top '+str(percent)+'%')
    ax[1][1].plot(N_loops, bot_AD_p, marker='.', color='orange', label='bottom '+str(percent)+'%')
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

    fig, ax = plt.subplots(1, 1, dpi=150)
    fig.suptitle('Testing Probability Value Certainty for Split on Environmental Property')
    ax.set_xlabel(r'$N_{loops}$')
    ax.set_ylabel(r'$std(p(N_{loops}))$')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.plot(N_loops, stds)
    ax.plot(N_loops, stds)
    comment = mpatches.Patch(edgecolor='black', facecolor='white', label='satellite parameter: ' + labelb)
    comment2 = mpatches.Patch(edgecolor='black', facecolor='white', label='environment parameter: ' + labela)
    fig.legend(handles=[comment, comment2], loc="center right", fontsize=8)
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
        [M31,M32,M33]])
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
def user_choice(ca_med, ba_med, ddim, dbright, dratio):
    """Asks user for inputted host properties to make a list of variables
    for the argument of sat_MW_comparison.
    """
    props = [ca_med, ba_med, ddim, dbright, dratio]
    # from table 3 Neuzil et al 2019
    MW = [0.163, 0.786, 1.699, 5.190, 1.699/5.190]
    labels = [r'$(c/a)_{median}$ ', r'$(b/a)_{median}$ ',
              r'$\Delta_{dim}$ ', r'$\Delta_{bright}$ ', r'$\Delta_{ratio}$']
    propsa = []
    MWa = []
    labelsa  = ''
    for i in range(len(props)):
        a = str(input("Choose properties by entering a number 0-4 corresponding to "
                      "ca_median, ba_median, ddim, dbright, dratio. "
                      "Enter DONE after final choice or ALL if you'd like all data considered: "))
        print('')
        if i == 0:
            while a == 'DONE':
                a = str(input('User must enter ALL or a number 0-4: '))
                print('')
            if a == 'ALL':
                Label = 'all environmental parameters'
                return props, MW, Label
        if i != 0:
            if a == 'DONE':
                break
            while a == 'ALL':
                a = str(input('User must enter DONE or a number 0-4: '))
                print('')
        if a != 'DONE' or 'ALL':
            a = int(a)
            propsa.append(props[a])
            MWa.append(MW[a])
            labelsa += labels[a]
    return propsa, MWa, labelsa


def histMW(vara, varb, ax, per, labelb, setax, upperx,            # vara: env, varb: sat
               uppery, bins, colors, xtitle, ytitle, leg):

    lower_lim = vara < np.percentile(vara, per)
    N_loops = 100  # change to 50k
    low_p_KS, p = empirical_KS(varb, varb[lower_lim], N_loops)
    low_p_AD = empirical_AD(varb, varb[lower_lim], N_loops)
    # each of these curves are CDFs
    ax.hist(varb[lower_lim], bins=bins, cumulative=True, density=True, range=(0, upperx), histtype='step'
            , color=colors[1], label='KS pvalue ' + str(round(low_p_KS,2)))
    ax.hist(varb[lower_lim], bins=bins, cumulative=True, density=True, range=(0, upperx), histtype='step'
            , color=colors[1], label='AD pvalue ' + str(round(low_p_AD,2)))
    ax.hist(varb, bins=bins, cumulative=True, density=True, range=(0, upperx), histtype='step', color=colors[2])
    ax.axis('square')
    if setax is True:
        ax.set_ylim(0, uppery)
        ax.set_xlim(0, upperx)
    if xtitle is True:
        ax.set_xlabel(labelb, fontsize=14)
    if ytitle is True:
        ax.set_ylabel('N_hosts', fontsize=14)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    if leg is True:
        ax.legend(loc="upper left", fontsize=11)
    elif leg is False:
        ax.legend(loc="lower right", fontsize=11)


def statistics_MW_rank(sat1, sat2, euc_dist, label1, label2, sim_components):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10), dpi=75, tight_layout=False)
    upperx = 1.0
    uppery = 1.0
    bins = 200
    per_list = [5, 10, 25]
    colors5 = ['indianred', 'indianred','black']
    colors10 = ['forestgreen', 'forestgreen', 'black']
    colors25 = ['purple', 'purple','black']
    colors = [colors5, colors10, colors25]
    ytitle = False
    xtitle = False
    setax = True
    leg = False
    i = 0
    for per in per_list:
        if i == 2:
            xtitle = True
        histMW(euc_dist, sat1, ax[i][0], per, label1,
               setax, upperx, uppery, bins, colors[i], xtitle, True, leg)
        histMW(euc_dist, sat2, ax[i][1], per, label2,
               setax, upperx, uppery, bins, colors[i], xtitle, ytitle, True)
        i += 1
    fig.suptitle('Histogram of Satellite Properties Split on  Similarity Percentiles to Milky Way',
                 y=0.97, fontsize=16)
    line1 = mlines.Line2D([], [], color='indianred', label='5% most similar')
    line3 = mlines.Line2D([], [], color='forestgreen', label='10% most similar')
    line5 = mlines.Line2D([], [], color='purple', label='25% most similar')
    line7 = mlines.Line2D([], [], color='black', label='all hosts')
    comment = mpatches.Patch(edgecolor='black', facecolor='white', label='AD: Anderson-Darling Test')
    comment2 = mpatches.Patch(edgecolor='black', facecolor='white', label='KS: Kolmogorov-Smyrnov Test')
    comment3 = mpatches.Patch(edgecolor='black', facecolor='white', label='similarity computed with: '+ sim_components)
    fig.legend(handles=[line1, line3, line5, line7, comment, comment2, comment3], fontsize=8)
    plt.subplots_adjust(wspace=0.2)
    plt.show()


# two of two ways to select subsamples
def sat_MW_comparison(props, all_props, MW_data, labels, all_labels, ca_rvir, ba_rvir, corotations):
    """This function ranks the catalog's host halos
    in similarity to the Milky Way.

    :param props: list of environmental properties by which to rank
    :param MW_data: environmental property values for the Milky Way
    """
    # normalize arrays with standard deviation, avoids outliers
    dist_arrays = []
    i = 0
    for prop in props:
        prop = np.copy(prop)
        MW_prop = MW_data[i]
        std = np.std(prop)  # a measure of the spread of a distribution
        prop /= std
        MW_prop /= std
        dist = np.square(prop - MW_prop)
        dist_arrays.append(dist)
        i += 1
    sum = 0
    for dist in dist_arrays:
        sum += dist
    euc_dist = np.sqrt(sum)  # using the euclidean distance equation
    most_similar_halo = np.amin(euc_dist)
    least_similar_halo = np.amax(euc_dist)
    most_index = np.argmin(euc_dist)
    least_index = np.argmax(euc_dist)
    print('')
    print('least distance:', most_similar_halo, 'most distance:', least_similar_halo)
    print('most similar halo index:', most_index, 'least similar halo index:', least_index)
    statistics_MW_rank(corotations, ca_rvir, euc_dist, label1=r'$f_{corotation}$',
                       label2=r'$(c/a)_{rvir}$', sim_components=labels)
    statistics_MW_rank(corotations, ba_rvir, euc_dist, label1=r'$f_{corotation}$',
                       label2=r'$(b/a)_{rvir}$', sim_components=labels)

    most_indices = np.argpartition(euc_dist,range(5))[:5]
    list_2d = []
    i = 0
    for index in most_indices:
        list = []
        for prop in all_props:
            list.append(prop[index])
        list_2d.append(list)
        """
        print('most similar ' + str(i), all_labels, list_2d[i])
        print('')
        """
        i += 1
    return euc_dist
    # paper test!! worked


# git clone "github link"
# or wget "url"
# leave buggy halo
# more combos of MW rank !!!

def KS_2D(sat1, sat2, env_list, labels, per):
    """
    Comparing 2d samples: ca_rvir and corotation
    using existing code
    """
    fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=75, tight_layout=False)
    ax[0][0].set_ylabel(r'$f_{corotation}$', fontsize=14)
    ax[1][0].set_ylabel(r'$f_{corotation}$', fontsize=14)
    ax[1][0].set_xlabel(r'$(c/a)_{rvir}$', fontsize=14)
    ax[1][1].set_xlabel(r'$(c/a)_{rvir}$', fontsize=14)
    fig.suptitle('2D Scatter of Satellite Properties Split on Environmental Parameter Percentiles')
    upperx = 0.9
    uppery = 0.6
    colors = ['red', 'limegreen', 'green', 'darkviolet', 'white', 'black']
    j = 0
    k = 0
    for i in range(len(env_list)):
        env_cut1 = env_list[i] < np.percentile(env_list[i], per)
        a = np.array([sat1, sat2])
        b = np.array([sat1[env_cut1], sat2[env_cut1]])
        _, p = KS2D.ks2d2s(a, b)
        if i == 1 :
            k = 1
        if i == 2:
            j = 1
            k = 0
        if i == 3:
            j = 1
            k = 1
        ax[j][k].scatter(sat1, sat2, marker='.', edgecolors=colors[-2], color=colors[-1], label='full satellite sample')
        ax[j][k].scatter(sat1[env_cut1], sat2[env_cut1],  marker='.', color=colors[0], label='KS pvalue ' + str(round(p,2)))
        ax[j][k].set_xlim(0, upperx)
        ax[j][k].set_ylim(0, uppery)
        ax[j][k].set_title('Split on lowest ' + str(per) + '% ' + labels[i])
        ax[j][k].legend(loc='best')
    plt.show()


def alignment(all_labels, all_props, props, MW_data, ca_rvir, minor_vir, minor_8, per):
    dist_arrays = []
    i = 0
    for prop in props:
        prop = np.copy(prop)
        MW_prop = MW_data[i]
        std = np.std(prop)  # a measure of the spread of a distribution
        prop /= std
        MW_prop /= std
        dist = np.square(prop - MW_prop)
        dist_arrays.append(dist)
        i += 1
    sum = 0
    for dist in dist_arrays:
        sum += dist

    euc_dist = np.sqrt(sum)
    lim = euc_dist < np.percentile(euc_dist, per)

    sat_minor = minor_vir[lim]
    env_minor = minor_8[lim]
    ca_rvir = ca_rvir[lim]

    indices = np.where(lim == True)[0]
    indices5 = [0, 1, 2, 3, 4, 5]
    """
    list_2d = []
    i = 0
    for index in indices:
        list = []
        for prop in all_props:
            list.append(prop[index])
        list_2d.append(list)
        print('most similar ' + str(i), all_labels, list_2d[i])
        print('')
        i += 1
    """
    differences = []
    for i in range(len(sat_minor)):
        cos_theta = (np.dot(sat_minor[i], np.reshape(env_minor[i], (3,1)))) / \
                    (np.linalg.norm(sat_minor[i]) * np.linalg.norm(env_minor[i]))
        differences.append(np.absolute(cos_theta[0][0]))
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    plt.suptitle('Difference Between Angles from Minor Axis at Virial and 8Mpc Radii')
    ax.scatter(ca_rvir, differences, marker='.', label= str(per)+'% most similar to MW')
    ax.set_xlabel(r'$c/a_{rvir}$')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_ylabel(r'$|cos(\theta)|$')
    plt.legend()
    plt.show()

def opening_angle(xi, yi, zi, xj, yj, zj):
    dot = (xi*xj) + (yi*yj) + (zi*zj)
    pos = np.where(dot > 0)[0]
    neg = np.where(dot < 0)[0]
    normi = ((xi**2) + (yi**2) + (zi**2))**0.5
    normj = ((xj**2) + (yj**2) + (zj**2))**0.5
    thetas = np.arccos(dot / (normi*normj))
    return thetas

# will underestimate p-value if it doesn't account for correlation of theta_opens;
# opening angles are NOT independent of one another (axis ratios are)
def lilliefors(host_ids, full_sample, subsample, N_loops):
    """ The null-hypothesis for the KT test is that the distributions are the same.
        Thus, the lower your p value the greater the statistical evidence you have to
        reject the null hypothesis and conclude the distributions are different.
    """
    count = 0
    all_sats = np.hstack(full_sample)
    full_sample = np.array(full_sample, dtype=object)
    og_stat, og_pvalue = stats.kstest(all_sats, np.hstack(subsample))
    for i in range(N_loops):
        num_points = len(subsample)
        ran_ids = np.random.choice(host_ids, num_points, replace=True)
        ran_hosts = full_sample[ran_ids]
        ran_sats = np.hstack(ran_hosts)
        emp_stat, emp_pvalue = stats.kstest(all_sats, ran_sats)
        if emp_stat > og_stat:
            count += 1
    updated_pvalue = count / N_loops
    return updated_pvalue, og_pvalue

def single_KS(full, sub):
    stat, p = stats.kstest(full, sub)
    return p

def sat_sep_plots(env, ax, per, labela, labelb, subtitle, xtitle, ytitle, sat_positions, hosts, L, Num, bins, dict, similarity_labels):
    sample = env < np.percentile(env, per)
    bins=bins
    all_thetas = []
    corot = []
    antirot = []
    samples = []
    corot_samples = []
    antirot_samples = []
    host_ids = []
    for h in range(len(hosts)):
        idx = np.arange(len(sat_positions[0][h]))
        i, j = idx, idx
        i_grid, j_grid = np.meshgrid(i, j)
        i_flat, j_flat = i_grid.flatten(), j_grid.flatten()
        valid_pair = j_flat > i_flat
        i_flat, j_flat = i_flat[valid_pair], j_flat[valid_pair]
        thetas = opening_angle(sat_positions[0][h][i_flat], sat_positions[1][h][i_flat], sat_positions[2][h][i_flat],
                               sat_positions[0][h][j_flat], sat_positions[1][h][j_flat], sat_positions[2][h][j_flat])
        all_thetas.append(thetas)
        Li, Lj = L[h][i_flat, :], L[h][j_flat, :]
        dots = np.sum(Li * Lj, axis=1)  # a value for each satellite, but want a single value for entire halo???
        corot.append(thetas[dots > 0])
        antirot.append(thetas[dots < 0])
        if sample[h] == True:
            samples.append(thetas)
            corot_samples.append(thetas[dots > 0])
            antirot_samples.append(thetas[dots < 0])
        host_ids.append(h)
    anglez = [np.hstack(all_thetas), np.hstack(corot), np.hstack(antirot), np.hstack(samples),
              np.hstack(corot_samples), np.hstack(antirot_samples)]
    centerz = []
    pdenz = []
    for a in anglez:
        N, theta_edges = np.histogram(a, range=(0, np.pi), bins=bins)
        bin_size = np.pi / bins
        p_density = N / np.sum(N) / bin_size
        theta_centers = (theta_edges[1:] + theta_edges[:-1]) / 2
        sin_theta = np.sin(theta_centers) / 2
        scaled_pdensity = p_density / sin_theta
        centerz.append(theta_centers)
        pdenz.append(scaled_pdensity)
    ps, _ = lilliefors(host_ids, all_thetas, samples, Num)
    pc, _ = lilliefors(host_ids, corot, corot_samples, Num)
    pa, _ = lilliefors(host_ids, antirot, antirot_samples, Num)
    ax.plot(centerz[0], pdenz[0], label=r'all $\theta_{open}$', color='blue')
    ax.plot(centerz[1], pdenz[1], label='corotating', color='red')
    ax.plot(centerz[2], pdenz[2], label='antirotating', color='purple')
    ax.plot(centerz[3], pdenz[3], label=str(per) + r' % MW similarity: $p$ = ' + str(round(ps, 2)), color='blue', ls='--')
    #ax.plot(centerz[3], pdenz[3], label='top ' + str(per) + r'% in ' + labela + ': $p$ = ' + str(round(ps, 2)))
    ax.plot(centerz[4], pdenz[4], label=r'corotating in ' + str(per) + '%: $p$ = ' + str(round(pc, 2)), color='red', ls=':')
    ax.plot(centerz[5], pdenz[5], label=r'antirotating in ' + str(per) + '%: $p$ = ' + str(round(pa, 2)), color='purple', ls='-.')
    ax.axhline(y=1, color='black', lw=0.8, linestyle='--', label='expected spherical dist')
    ax.axhline(y=0, color='white', lw=0.8, linestyle='--', label='similarity computed on: ' + str(similarity_labels))

    # each of these curves are CDFs
    ax.set_ylim(0.8, 1.6)
    ax.set_aspect(4)
    if subtitle is True:
        ax.set_title('split on ' + labela, fontsize=10)
    if xtitle is True:
        ax.set_xlabel(labelb, fontsize=10)
    if ytitle is True:
        ax.set_ylabel(r'$N_{\theta_{open}}\ /\ N_{tot}\ /\ d(\theta_{open})/\ sin(\theta)$', fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="best", fontsize=5)
    string = str(per) + '% ' + labela
    dict[string] = {r'$p$-values': {'all-sample': ps, 'corot-sample': pc, 'antirot-sample': pa}}
    return dict


def sat_sep(sat_positions, L, hosts, euc_dist, similarity_labels, env_props, env_labels):
    """
    fig, ax = plt.subplots(3,4,figsize=(10, 10), dpi=75, tight_layout=False)
    i = 0
    bins=10
    per_list = [5, 10, 25]
    ytitle = False
    xtitle = False
    labelb = r'Opening angle $\theta_{open}$ [rad]'
    title = True
    dict = {}
    N1 = 10000
    for per in per_list:
        if i == 0:
            title = True
        if i == 1 or i == 2:
            title = False
        if i == 2:
            xtitle = True
        sat_sep_plots(env_props[0], ax[i][0], per, env_labels[0],
                labelb, title, xtitle, True, sat_positions, hosts, L, N1, bins, dict)
        sat_sep_plots(env_props[1], ax[i][1], per, env_labels[1],
                labelb, title, xtitle, ytitle, sat_positions, hosts, L, N1, bins, dict)
        sat_sep_plots(env_props[2], ax[i][2], per, env_labels[2],
                labelb, title, xtitle, ytitle, sat_positions, hosts, L, N1, bins, dict)
        sat_sep_plots(env_props[3], ax[i][3], per, env_labels[3],
                labelb, title, xtitle, ytitle, sat_positions, hosts, L, N1, bins, dict)
        i += 1
    print(dict)
    fig.suptitle('CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_L63 Split on Single Environmental Parameters')
    plt.show()
    """

    fig, ax = plt.subplots(3,4,figsize=(10, 10), dpi=75, tight_layout=False)
    i = 0
    bins=10
    per_list = [5, 10, 25]
    ytitle = False
    xtitle = False
    labelb = r'Opening angle $\theta_{open}$ [rad]'
    title = False
    dict = {}
    N1 = 10000
    for per in per_list:
        if i == 1 or i == 2:
            title = False
        if i == 2:
            xtitle = True
        sat_sep_plots(euc_dist[0], ax[i][0], per, env_labels[0],
                labelb, title, xtitle, True, sat_positions, hosts, L, N1, bins, dict, similarity_labels[0])
        sat_sep_plots(euc_dist[1], ax[i][1], per, env_labels[1],
                labelb, title, xtitle, ytitle, sat_positions, hosts, L, N1, bins, dict, similarity_labels[1])
        sat_sep_plots(euc_dist[2], ax[i][2], per, env_labels[2],
                labelb, title, xtitle, ytitle, sat_positions, hosts, L, N1, bins, dict, similarity_labels[2])
        sat_sep_plots(euc_dist[3], ax[i][3], per, env_labels[3],
                labelb, title, xtitle, ytitle, sat_positions, hosts, L, N1, bins, dict, similarity_labels[3])
        i += 1
    print(dict)
    """
    line1 = mlines.Line2D([], [], color='indianred', label='top 5%')
    line3 = mlines.Line2D([], [], color='forestgreen', label='top 10%')
    line5 = mlines.Line2D([], [], color='crimson', label='top 25%')
    line7 = mlines.Line2D([], [], color='black', label='all hosts')
    fig.legend(handles=[line1, line3, line5, line7], fontsize=6)"""
    fig.suptitle('CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_L63 selected on Percentiles of Similarity to MW')
    plt.show()

    """bins = 10
    per = 10
    labelb = r'Opening angle $\theta_{open}$ [rad]'
    title = True
    fig, ax = plt.subplots(figsize=(10, 10), dpi=75, tight_layout=False)
    sat_sep_plots(env_props[2], ax, per, env_labels[2],
                  labelb, title, True, True, sat_positions, hosts, L, 10000, bins)
    fig.suptitle(r'CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_L63 Split on Single Environmental Parameter: $\Delta_{dim}$')
    plt.show()
    all_thetas2 = []
    corot2 = []
    antirot2 = []
    samples2 = []
    corot_samples2 = []
    antirot_samples2 = []
    host_ids2 = []
    # use euc_dist for MW comparison, and env_props for single parameter
    sample = env_props[2] < np.percentile(env_props[2], per)   ### ddim
    for h in range(len(hosts)):
        idx = np.arange(len(sat_positions[0][h]))
        i, j = idx, idx
        i_grid, j_grid = np.meshgrid(i, j)
        i_flat, j_flat = i_grid.flatten(), j_grid.flatten()
        valid_pair = j_flat > i_flat
        i_flat, j_flat = i_flat[valid_pair], j_flat[valid_pair]
        thetas = opening_angle(sat_positions[0][h][i_flat], sat_positions[1][h][i_flat], sat_positions[2][h][i_flat],
                               sat_positions[0][h][j_flat], sat_positions[1][h][j_flat], sat_positions[2][h][j_flat])
        all_thetas2.append(thetas)
        Li, Lj = L[h][i_flat, :], L[h][j_flat, :]
        dots = np.sum(Li * Lj, axis=1)  # a value for each satellite, but want a single value for entire halo???
        corot2.append(thetas[dots > 0])
        antirot2.append(thetas[dots < 0])
        if sample[h] == True:
            samples2.append(thetas)
            corot_samples2.append(thetas[dots > 0])
            antirot_sample2s.append(thetas[dots < 0])
        host_ids2.append(h)
    labelz2 = ['all', 'corot', 'antirot', 'similarity percentile', 'similarity corot', 'similarity antirot']
    anglez2 = [np.hstack(all_thetas2), np.hstack(corot2), np.hstack(antirot2), np.hstack(samples2),
              np.hstack(corot_samples2), np.hstack(antirot_samples2)]
    centerz2 = []
    pdenz2 = []
    for a in anglez2:
        N, theta_edges = np.histogram(a, range=(0, np.pi), bins=bins)
        bin_size = np.pi / bins
        p_density = N / np.sum(N) / bin_size
        theta_centers = (theta_edges[1:] + theta_edges[:-1]) / 2
        sin_theta = np.sin(theta_centers) / 2
        scaled_pdensity = p_density / sin_theta
        centerz2.append(theta_centers)
        pdenz2.append(scaled_pdensity)

    N_loops = 10000
    ps, _ = lilliefors(host_ids2, all_thetas2, samples2, N_loops)
    pc, _ = lilliefors(host_ids2, corot2, corot_samples2, N_loops)
    pa, _ = lilliefors(host_ids2, antirot2, antirot_samples2, N_loops)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=75, tight_layout=False)
    ax.plot(centerz2[0], pdenz2[0], label=r'all $\theta_{open}$')
    ax.plot(centerz2[1], pdenz2[1], label='corotating')
    ax.plot(centerz2[2], pdenz2[2], label='antirotating')
    ax.plot(centerz2[3], pdenz2[3], label=str(per)+r' % MW similarity: $p$ = ' + str(round(ps, 2)))
    #ax.plot(centerz2[3], pdenz2[3], label='top ' + str(per) + r'% in ' + env_labels[0]+': $p$ = ' + str(round(ps, 2)))
    ax.plot(centerz2[4], pdenz2[4], label=r'corotating in '+str(per)+'%: $p$ = ' + str(round(pc, 2)))
    ax.plot(centerz2[5], pdenz2[5], label=r'antirotating in '+str(per)+'%: $p$ = ' + str(round(pa, 2)))
    ax.plot()
    ax.axhline(y=1, color='black', lw=0.8, linestyle='--', label='expected spherical dist')
    ax.axhline(y=0, color='white', lw=0.8, linestyle='--', label='bins: '+str(bins))
    ax.axhline(y=0, color='white', lw=0.8, linestyle='--', label='similarity computed on: '+str(similarity_labels))
    #fig.suptitle('CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_L63 selected on Percentiles of Similarity to MW')
    fig.suptitle(r'CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_L63 Split on Single Environmental Parameter: $\Delta_{dim}$')
    ax.set_xlabel(r'Opening angle $\theta_{open}$ [rad]')
    ax.set_ylabel(r'$N_{\theta_{open}}\ /\ N_{tot}\ /\ d(\theta_{open})/\ sin(\theta)$')
    ax.set_ylim(0.75, np.pi)
    ax.axis('square')
    ax.legend()
    plt.show()"""

    print('done')


def random_sphere(N):
    """ random_sphere returns the azimuthal angle, phi, and polar angle, theta
    of N points generated uniformly at random over the surface of a sphere.
    """
    # See https://mathworld.wolfram.com/SpherePointPicking.html for the
    # algorithm. (Basically just the Inverse Transfer Method.)
    phi = 2 * np.pi * random.random(N)
    theta = np.arccos(2 * random.random(N) - 1)
    return phi, theta

def spherical_to_cartesian(phi, theta, r):
    """ spherical_to_cartesian converts spherical coordiantes to cartesian
    coordinates. Here, phi is the azimuthal angle, theta is the polar angle, and
    r is the radius.
    """
    # See https://mathworld.wolfram.com/SphericalCoordinates.html, but note the
    # difference in convention.
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z

def random_ball(N, R):
    """ random_ball returns a N points (x, y, z) generated uniformly at
    random from within a ball of radius r.
    """
    phi, theta = random_sphere(N)
    # Inverse Transform Method, See
    # http://www.columbia.edu/~ks20/4404-Sigman/4404-Notes-ITM.pdf
    r = random.random(N) ** (1.0 / 3) * R
    return spherical_to_cartesian(phi, theta, r)


def random_ellipsoid(N, a, b, c):
    """ random_ellipsoid returns N points generated uniformly at random inside
    an ellipsoid. The axes of the ellipsoid are given by (a, b, c) which
    are aligned with the x, y, and z axes, respectively.
    """
    x, y, z = random_ball(N, 1.0)
    return x * a, y * b, z * c

"""
N = 500
def test_sep():

    #x, y, z = random_ellipsoid(N, 25.904750799398368, 21.681969785452043, 19.293134199600996)
    x, y, z = random_ellipsoid(N, 1, 1, 0.5)
    #x, y, z = random_ball(N, 1.0)
    #print('point displacements', x,y,z)
    d = []
    for k in range(len(x)):
        x1 = x[k]
        y1 = y[k]
        z1 = z[k]
        d.append([x1,y1,z1])
    d = [d]
    #d = [[[0,0,1],[2,0,0],[0,0,-2]]]
    sat_sep(d)
"""

def test_N_sats():
    # x, y, z are arrays of position values for each random point in the
    # ellipsoid, relative to its center
    fig, ax = plt.subplots(figsize=(10, 10), dpi=75)
    N_points = [5, 10, 15, 20]
    N_realizations = 100
    ca_exp = 1
    ca_2d = []
    ca_stds = []
    N_2d = []
    for n in N_points:
        ax.plot(n, ca_exp, color='red', marker='x')
        ca_point = []
        N_point = []
        for i in range(N_realizations):
            x, y, z = random_ellipsoid(n, 1, 1, 1)
            ca, ba, _, _ = get_axes(x, y, z, i)
            ca_point.append(ca/ca_exp)
            N_point.append(n)
        ca_2d.append(ca_point)
        ca_stds.append(np.std(ca_point))
        N_2d.append(N_point)

    ax.scatter(N_2d, ca_2d)
    plt.errorbar(N_2d, ca_2d, ecolor='pink', yerr=ca_stds, xerr=None, ls='none')
    ax.set_xlabel(r'$N_{points}$')
    ax.set_ylabel(r'$(c/a)\ / \ (c/a_{expected})$')
    fig.suptitle(r'Axis Ratio Recovery for Various $N_{sats}$ Done With ' + str(N_realizations) + ' Point Realizations')
    plt.show()
    """
    x, y, z = random_ellipsoid(N, 1, 1, 0.5)
    ca_exp = 0.5
    x, y, z = random_ellipsoid(N, 1, 0.5, 0.5)
    ca_exp = 0.5
    x, y, z = random_ellipsoid(N, 1, 1, 0.75)
    ca_exp = 0.75
    x, y, z = random_ellipsoid(N, 1, 0.75, 0.75)
    ca_exp = 0.75
    """




if __name__ == '__main__':
    main()