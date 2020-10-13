# adding stuff
import minh
import numpy as np
import matplotlib.pyplot as plt
import math

"""
 This code works with a complete DM halo simulation, minh.
 A distance-based search for subhaloes within a single chosen
 host halo is done. The axis ratio of this host is computed 
 and checked against the axis ratio values provided in the 
 simulation's existing data.
"""
L = 125
mass = 1
def main():
    x, y, z, mvir, rvir, id, upid, b_to_a, c_to_a = get_file_data()
    host_index, host_radius, host_unique_id, host_upid, host_mass = \
        host_halo(id, upid, rvir, mvir)
    sub_x, sub_y, sub_z = get_halo_distances(host_index, host_radius,
                                             id, upid, x, y, z)
    a_len, b_len, c_len = get_axes(sub_x, sub_y, sub_z)
    c_a_ratio = c_len / a_len
    b_a_ratio = b_len / a_len
    print('Axis ratio c:a is '+str(c_a_ratio))
    print('Axis ratio b:a is '+str(b_a_ratio))
    correlation(c_to_a, b_to_a, host_index, a_len, b_len, c_len)


def get_file_data():
    """
    comment
    """
    f=minh.open('hlist_1.00000.minh')
    x, y, z, mvir, rvir, id, upid, b_to_a, c_to_a = f.read(['x','y','z',
                            'mvir','rvir','id','upid','b_to_a','c_to_a'])
    print(len(x), len(mvir), len(rvir), len(b_to_a))  #all outputs = 672778
    return x, y, z, mvir, rvir, id, upid, b_to_a, c_to_a


def host_halo(id, upid, rvir, mvir):
    """
    Parses through the data file and saves the host-halo
    ids into a list, from here the user can pick which
    one they want to receive the axis-ratio for.
    """
    host_index = int(input('Enter host halo index: '))
    host_unique_id = id[host_index]
    host_upid = upid[host_index]   # -1 if host, big num if subhalo
    host_radius = rvir[host_index]
    host_mass = mvir[host_index]
    return host_index, host_radius, host_unique_id, host_upid, host_mass



def find_subhaloes(host_index, host_radius, id, upid, distance):
    subhaloes = np.where(distance*1000 <= host_radius)
    # check against upids
    subhalo_indices = np.where(id[host_index] == upid)
    subhaloes = subhaloes[0][1:]
    bool = np.all(subhaloes == subhalo_indices[0])
    print('Subhaloes match? ' + str(bool))
    return subhaloes


### L is still 125 for large simulation
def check_boundary(x, y, z):
    """
    Using the halo displacement matrices for x,y,z
    check if the each subhalo position for x,y,z is a
    sensible distance from its host. If it's not, the halo
    has been "split" to the other side of the simulation box,
    relative to the host halo under study,
    and this must be corrected for.
    """
    list = [x, y, z]
    for position in list:
        # subhalo split and on right side, host left
        too_big = np.where(position > (L/2))
        # subhalo split and on left side, host right
        too_small = np.where(position < (-L/2))
        position[too_big] -= L
        position[too_small] += L
    return  x, y, z


def get_halo_distances(host_index, host_radius, id, upid, x, y, z):
    host_x, host_y, host_z = x[host_index], y[host_index], z[host_index]
    # finding DISPLACEMENTS between all objects and this host halo
    x -= host_x
    y -= host_y
    z -= host_z
    # fixing these displacements
    x, y, z = check_boundary(x, y, z)
    # finding the DISTANCES between all objects and this host halo
    distance = np.sqrt(x**2 + y**2 + z**2)
    # finding which distances are contained in host radius: subhaloes
    subhaloes = find_subhaloes(host_index, host_radius, id, upid, distance)
    sub_x = x[subhaloes]
    sub_y = y[subhaloes]
    sub_z = z[subhaloes]
    return sub_x, sub_y, sub_z



def make_inertia_tensor():
    """
    Create an empty 3x3 matrix to store our inertia tensor
    values.
    """
    empty_inertia_tensor = np.zeros((3, 3))
    return empty_inertia_tensor


def populate_inertia_tensor(inertia_tensor, sub_x, sub_y, sub_z):
    """
    Moments and Products of Inertia about various axes:
        Ixx = sum[(y^2 + z^2) * mass]
        Iyy = sum[(x^2 + z^2) * mass]
        Izz = sum[(x^2 + y^2) * mass]
        Ixy = Iyx = -sum[x * y * mass]
        Iyz = Izy = -sum[y * z * mass]
        Ixz = Izx = -sum[x * z * mass]

    We use this matrix to determine the moment of inertia for an
    arbitrarily shaped object, characterizing its shape.
    """
    inertia_tensor[0][0] = np.sum(((sub_y**2) + (sub_z**2)) * mass)
    inertia_tensor[0][1] = -np.sum((sub_x * sub_y * mass))
    inertia_tensor[0][2] = -np.sum((sub_x * sub_z * mass))
    inertia_tensor[1][0] = -np.sum((sub_x * sub_y * mass))
    inertia_tensor[1][1] = np.sum(((sub_x ** 2) + (sub_z ** 2)) * mass)
    inertia_tensor[1][2] = -np.sum((sub_y * sub_z * mass))
    inertia_tensor[2][0] = -np.sum((sub_x * sub_z * mass))
    inertia_tensor[2][1] = -np.sum((sub_y * sub_z * mass))
    inertia_tensor[2][2] = np.sum(((sub_x ** 2) + (sub_y ** 2)) * mass)
    return inertia_tensor


def compute_e_values(inertia_tensor):
    """
    Function computes the eigenvalues and right eigenvectors
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


def get_axes(x_arr, y_arr, z_arr):
    """
    Returns the axis lengths largest to smallest of the halo
    delineated as a,b,c.
    """
    empty_inertia_tensor = make_inertia_tensor()
    inertia_tensor = populate_inertia_tensor(empty_inertia_tensor,
                                             x_arr, y_arr, z_arr)
    evalues = compute_e_values(inertia_tensor)
    axis_a_length, axis_b_length, axis_c_length = convert_to_length(
        evalues)
    return axis_a_length, axis_b_length, axis_c_length


def correlation(c_to_a, b_to_a, host_index, a_len, b_len, c_len):
    """
    Between my code's axis ratio outputs and the simulation's data.
    """
    expected_ca = c_to_a[host_index]
    expected_ba = b_to_a[host_index]
    fig, (ax1, ax2) = plt.subplots(2, 1, dpi=100)
    fig.suptitle('Axis Ratio Correlations with Subhalo Abundance')
    ax1.set_title('c:a Axis Ratio')
    ax2.set_title('b:a Axis Ratio')
    ax2.set_xlabel(r'$log(N_{sh})$')
    ax1.set_ylabel('Mean Fractional Error')
    ax2.set_ylabel('Mean Fractional Error')
    N_list = np.logspace(1, 4, num=50, endpoint=True, base=10.0,
                         dtype=int, axis=0).tolist()
    num_N = int(input('How many N iterations? '))
    error_ca_arr = np.zeros(len(N_list))
    error_ba_arr = np.zeros(len(N_list))
    for i in range(num_N):
        error_ca_list = []
        error_ba_list = []
        for N in N_list:
            measured_ca = c_len / a_len
            measured_ba = b_len / a_len
            error_ca_list.append(np.absolute((measured_ca - expected_ca))
                                 / expected_ca)
            error_ba_list.append(np.absolute((measured_ba - expected_ba))
                                 / expected_ba)
        error_ca_arr += np.asarray(error_ca_list)
        error_ba_arr += np.asarray(error_ba_list)
    error_ca_arr /= num_N
    error_ba_arr /= num_N
    ax1.plot(N_list, error_ca_arr, '.r-', markersize=1.5, linewidth=0.5)
    ax2.plot(N_list, error_ba_arr, '.b-', markersize=1.5, linewidth=0.5)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()