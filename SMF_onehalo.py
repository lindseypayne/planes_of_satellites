import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal
# change???
"""
Making a histogram of how many subhaloes are in a certain mass bin for the 
largest host halo in halo_data.txt file. Both axes will be logarithmic. 

Basically looking at the subhalo mass function curve for a large host.
"""

filename = 'halo_data.txt'
bins = 50

def main():
    # creating canvas for histogram
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100, tight_layout=True)
    fig.suptitle('Histogram of the SMF', x=0.58, fontsize=16)
    ax.set_xlabel(r'$log\left( \dfrac{m_{sh}}{M_{z_0}}\right)$', fontsize=14)
    ax.set_ylabel(r'$log\left[ \dfrac{dN}{dln\left(\dfrac{m_{sh}}{M_{z_0}}\right)}'
                  r'\right]$', fontsize=14)
    ax.axis("equal")

    # calling data array and normalizing x-axis
    subhalo_masses, host_mass = get_halo_masses()
    subhalo_masses = subhalo_masses / host_mass
    normalization = (np.log10(subhalo_masses.max()) -
                     np.log10(subhalo_masses.min())) / bins

    # plot interpolated histogram curve of "centered edge" data
    center, n = find_bin_centers(subhalo_masses)
    host_mass_simplified = '%.2E' % Decimal(str(host_mass))
    plt.plot(10**center, n/normalization, label="host mass = " +
             str(host_mass_simplified) + r' $\dfrac{M_{\odot}}{h}$')
    plt.legend(loc="upper right")
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def get_halo_masses():
    """
    This function reads through the data file and finds the host halo with
    the largest mass. Save this host's mass data and its subhalo mass data
    into an array.

    Columns of interest from text file:
        0: Index (number assigmnent, a means of naming) of HOST halo
        3: Mass of each halo/subhalo
    """

    array = np.loadtxt(filename, usecols=(0, 3)).T
    all_ids = array[0]
    mass_arr = array[1]

    max_host_mass = np.amax(mass_arr)
    host_index = np.where(mass_arr == max_host_mass)  # array index
    host_id = all_ids[host_index]                     # file id

    subhalo_indices = np.where(all_ids == host_id)
    subhalo_indices = np.delete(subhalo_indices, 0)   # delete host
    subhalo_mass_arr = mass_arr[subhalo_indices]

    return subhalo_mass_arr, max_host_mass


def find_bin_centers(subhalo_masses):
    # array of counts in each bin, array of edge values
    n, edges = np.histogram(np.log10(subhalo_masses), bins=bins,
                            range=(np.log10(subhalo_masses.min()),
                                   np.log10(subhalo_masses.max())))
    left = edges[:-1]  # pick out right edge of every bin
    right = edges[1:]  # pick out left edge of every bin
    center = []
    for i in range(len(left)):
        elem = (left[i] + right[i]) / 2
        center.append(elem)
    center = np.asarray(center)
    return center, n




if __name__ == '__main__':
    main()