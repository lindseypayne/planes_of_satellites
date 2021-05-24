import numpy as np
import numpy.linalg as linalg
from tabulate import tabulate
from env_stats import file_data, make_grid, check_boundary, get_axes

mass = 1
R = 8*0.7
V_sim = (4*np.pi*(R**3)) / 3
sigma = 0


def main():
    """Prints MW analog satellite/subhalo data in chosen
    DM-only simulation.
    """
    f, x, y, z, mvir, rvir, id, upid, M_B, mpeak, vx, vy, vz, target_ids = file_data("L63_hlist_1.00000.minh")
    Rvir, grid, points = make_grid(f, rvir, mvir, x, y, z)
    rows = []
    i = 1
    for id in target_ids[0]:
        host_point = points[id]
        host_Rvir = rvir[id]

        sub_x, sub_y, sub_z, sub_idx = grid.retrieve_tagged_members(host_point, host_Rvir).T
        sub_dx = sub_x - host_point[0]
        sub_dy = sub_y - host_point[1]
        sub_dz = sub_z - host_point[2]
        sub_dx, sub_dy, sub_dz = check_boundary(sub_dx, sub_dy, sub_dz, f.L)

        sub_vx = vx[sub_idx] - vx[id]
        sub_vy = vy[sub_idx] - vy[id]
        sub_vz = vz[sub_idx] - vz[id]

        sub_masses = mvir[sub_idx]
        N_sats = len(sub_idx)
        ratios_rvir, _, _ = get_axes(sub_dx, sub_dy, sub_dz)

        corotation, L_vec = new_tensor(sub_dx, sub_dy, sub_dz, sub_vx, sub_vy, sub_vz)

        ### include sub positions, ids, masses, velocities????
        cols = [i, id, N_sats, ratios_rvir[0], ratios_rvir[1], corotation, L_vec]
        rows.append(cols)
        i += 1
    print(tabulate(rows, headers=["# MW analogs", "host index", "# sats", "ca_rvir", "ba_rvir", "corotation (0-1)", "ang momentum vector"]))



def rotation_momentum(sub_dx, sub_dy, sub_dz, vx, vy, vz, inertia_tensor, minor_ax):
    """Find the angular momentum of each satellite rotating around a host's minor axis.
    If satellites have all positive or all negative momenta they are corotating together.
    This function then computes a statistic for how much corotation each host has.
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
    corotation = np.abs(pos_ratio - neg_ratio)
    return corotation, L_vec


# difference has to do with dividing by np.sum(mass) ????
def new_tensor(sub_dx, sub_dy, sub_dz, vx, vy, vz):
    """Computes the eigenvalues and vectors of a given host, its
    minor axis vector, calls rotation_momentum, and returns the
    host's corotation statistic and angular momentum vector.
    """
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
    evalues, evectors = np.linalg.eig(inertia_tensor)
    min_evalue = np.where(evalues == evalues.min())[0]
    minor_ax = np.reshape(evectors[min_evalue], (1,3))
    return rotation_momentum(sub_dx, sub_dy, sub_dz, vx, vy, vz, inertia_tensor, minor_ax)


if __name__ == '__main__':
    main()