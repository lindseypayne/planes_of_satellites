import numpy as np
import matplotlib.pyplot as plt

file = 'halo_data.txt'
bins = 75

"""
This program references a file containing property data for a number of 
host dark matter haloes and their respective subhaloes. Using this data,
we compute the joint subhalo mass function (SMF) of all host haloes in 
the file; this function is plotted as a histogram. A single curve for 
the average count of subhaloes per mass bin over all host haloes.

This attempts to reproduce the plots on page 7 of Giocoli's 2010 paper
titled "The Substructure Hierarchy in Dark Matter Haloes"

EVERY SMF SHOULD HAVE SAME NUMBER OF BINS AND SAME BIN WIDTHS/RANGES, 
SAME CENTER, BUT DIFFERENT COUNTS PER MASS BIN. I WANT TO GET THE SMF 
FOR EVERY SUBHALO ARRAY AND THEN TAKE THE AVERAGE AND PLOT THIS HISTOGRAM
"""

def main():
    # creating canvas for histogram
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100, tight_layout=True)
    fig.suptitle('Histogram of the Joint SMF', x=0.58, fontsize=16)
    ax.set_xlabel(r'$log\left(m_{sh} / M_{host}\right)$',
                  fontsize=14)
    ax.set_ylabel(r'$log[dN / dln(m_{sh} / M_{host})(N_{host})]$', fontsize=14)
    ax.axis("equal")

    total_hosts, host_halo_ids, mass_arr, all_ids = get_host_haloes()
    count_list = []
    # store each host's subhalo mass bin count into list of count arrays
    for i in range(total_hosts):
        sub_masses, host_mass = get_subhalo_mass(host_halo_ids, mass_arr,
                                                 all_ids, i)
        # checked and all centers are the same, now need to average counts
        center, n = bin_centers_and_counts(sub_masses)
        count_list.append(n)

    # add all counts in each bin so there's one array of counts
    for i in range(len(count_list)):
        if i == 0:
            this_count = count_list[i]
        else:
            this_count = np.add(previous_count, count_list[i])
        previous_count = this_count
    # divide by total number of host haloes to get average/joint SMH
    normalized_count = this_count / total_hosts
    # got log10(max, min) values from find_bin_centers testing below
    normalization1 = ((0 - -5) / bins)

    plt.plot(10 ** center, normalized_count / normalization1)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.legend(['bins = ' + str(bins)], loc="upper right")
    plt.show()


def get_host_haloes():
    array = np.loadtxt(file, usecols=(0, 3)).T
    all_ids = array[0]
    mass_arr = array[1]

    total_haloes = len(all_ids)
    host_halo_ids = []  # stores first instance of new id in a list
    for i in range(total_haloes):
        previous_halo = all_ids[i - 1]
        current_halo = all_ids[i]
        if previous_halo != current_halo and i < total_haloes - 1:
            host_halo_ids.append(current_halo)
    total_hosts = len(host_halo_ids)

    return total_hosts, host_halo_ids, mass_arr, all_ids


def get_subhalo_mass(host_halo_ids, mass_arr, all_ids, i):
    host_and_sub_indices = np.where(host_halo_ids[i] == all_ids)
    host_and_sub_masses = mass_arr[host_and_sub_indices]
    host_mass = host_and_sub_masses[0]
    sub_masses = host_and_sub_masses[1:]
    """
    normalize each subhalo by host halo mass
    this gives a general fraction of the whole
    rather than an absolute mass
    """
    sub_masses = sub_masses / host_mass
    return sub_masses, host_mass


# fix naming
def bin_centers_and_counts(sub_masses):
    """
    how I checked for proper hist range for all subhaloes
    print(np.log10(sub_masses.min()))
    print(np.log10(sub_masses.max()))
    """
    # array of counts in each bin, array of edge values
    n, edges = np.histogram(np.log10(sub_masses), bins=bins,
                            range=(-5, 0))
    left = edges[:-1]  # pick out right edge of every bin
    right = edges[1:]  # pick out left edge of every bin
    # array operations are vectorized
    center = (left + right) / 2
    """
    center = np.zeros(len(left))
    for i in range(len(left)):
        elem = (left[i] + right[i]) / 2
        center[i] = elem
    """
    return center, n


if __name__ == '__main__':
    main()