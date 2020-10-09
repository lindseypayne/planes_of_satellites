"""
This code finds the axis-ratio of a specified host halo and
also tests the accuracy of this code by plotting the
fractional error between a known axis-ratio and the calculated
one, as a function of N number of samples (i.e. subhaloes).
"""

### for boundary tests: check halo 18, 30 ###
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.random as random

infile = 'halo_data.txt'
mass = 1
L = 125


def main():
    all_ids, radius, x_coords, y_coords, z_coords = get_file_data()
    chosen_host_id = get_host_halo(all_ids)
    sub_x, sub_y, sub_z, host_radius = get_subhalo_coords(chosen_host_id,
                          all_ids, x_coords, y_coords, z_coords, radius)
    check_boundary(sub_x, sub_y, sub_z)
    a_len, b_len, c_len = get_axes(sub_x, sub_y, sub_z)
    c_a_ratio = c_len / a_len
    b_a_ratio = b_len / a_len
    print('Axis ratio c:a is '+str(c_a_ratio))
    print('Axis ratio b:a is '+str(b_a_ratio))
    correlation()


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


def get_file_data():
    """
    self-explanatory
    """
    array = np.loadtxt(infile, usecols=(0, 4, 8, 9, 10)).T
    all_ids = array[0]
    radius = array[1]
    x_coords = array[2]
    y_coords = array[3]
    z_coords = array[4]
    return all_ids, radius, x_coords, y_coords, z_coords


def get_host_halo(all_ids):
    """
    Parces through the data file and saves the host-halo
    ids into a list, from here the user can pick which
    one they want to receive the axis-ratio for.
    """
    total_haloes = len(all_ids)
    host_halo_ids = []  # stores first instance of new id in a list
    for i in range(total_haloes):
        previous_halo = all_ids[i - 1]
        current_halo = all_ids[i]
        if previous_halo != current_halo and i < total_haloes - 1:
            host_halo_ids.append(current_halo)
    total_hosts = len(host_halo_ids)
    pick_host = int(input('Choose a number between 0 and ' +
                          str(total_hosts - 1) + ': '))
    chosen_host_id = host_halo_ids[pick_host]
    return chosen_host_id


def get_subhalo_coords(chosen_host_id, all_ids, x, y, z, radii):
    host_and_sub_indices = np.where(chosen_host_id == all_ids)
    all_x, all_y, all_z = x[host_and_sub_indices], y[host_and_sub_indices], \
                          z[host_and_sub_indices]
    all_radii = radii[host_and_sub_indices]
    host_radius = all_radii[0]
    host_x, host_y, host_z = all_x[0], all_y[0], all_z[0]
    sub_x, sub_y, sub_z = all_x[1:], all_y[1:], all_z[1:]

    """
    Adjust subhalo positions so they're relative to center of host.
    Coordinate will be negative if subhalo center is positioned to 
    the left of the host center.
    """
    sub_x -= host_x
    sub_y -= host_y
    sub_z -= host_z
    return sub_x, sub_y, sub_z, host_radius


# TEST
def check_boundary(sub_x, sub_y, sub_z):
    """
    Using the subhalo "difference" matrices for x,y,z
    check if the each subhalo position for x,y,z is a
    sensible distance from its host. If it's not, the subhalo
    has been "split" to the other side of the simulation box,
    and this must be corrected for.
    """
    list = [sub_x, sub_y, sub_z]
    for position in list:
        # subhalo split and on right side, host left
        too_big = np.where(position > (L/2))
        # subhalo split and on left side, host right
        too_small = np.where(position < (-L/2))
        position[too_big] -= L
        position[too_small] += L


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


def correlation():
    """
    This function is a test of this code above. It takes in some arbitrary
    axis lengths and a randomly generated number of N points, representative
    of N subhaloes. An ellipsoid is therefore generated for my code to
    determine the axis ratios of. The expected ratio vs. my code's calculated
    ratio is compared and plotted against different N. This fractional error is
    taken for many iterations, of this N amount of sample points for each N value,
    and averaged.

    It is expected that the error at large N will be smaller than at small N.
    """
    a = 1
    b = 0.5
    c = 0.25
    expected_ca = c / a
    expected_ba = b / a
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
            x, y, z = random_ellipsoid(N, a, b, c)
            a_len, b_len, c_len = get_axes(x, y, z)
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