import numpy as np
import matplotlib.pyplot as plt
import palette
from palette import pc

def plot_circle(x0, y0, r, color, lw=2):
    """ plot_circle plots a circle with a given radius and color centered on
    the coordinates (x0, and y0).
    """
    theta = np.linspace(0, 7, 200)  # Overshoot 2*pi to prevent gaps.

    x = x0 + r * np.cos(theta)  # center of circle is x0,y0
    y = y0 + r * np.sin(theta)

    plt.plot([x0], [y0], ".", c=color)
    plt.plot(x, y, lw=lw, c=color)

def plot_subhaloes(mpeak, limit, rvir, vec):
    """ plot_subhaloes plots all subhaloes with mpeak > limit. Subhaloes are
    plotted as circles with radius rvir with their positions relative to the
    halo center. Only the (x, y) projection is plotted.
    """
    ok = mpeak > limit   #
    mpeak, rvir, vec = mpeak[ok], rvir[ok], vec[ok]

    for i in range(len(mpeak)):
        if i == 0:
            c, lw = "k", 3    # color and linewidth
        else:
            c, lw = pc(), 2
        dx, dy = vec[i, 0] - vec[0, 0], vec[i, 1] - vec[0, 1],
        plot_circle(dx, dy, rvir[i], c, lw=lw)

    # Leave a small buffer around each halo.
    spacing = 1.2
    plt.xlim(-rvir[0] * spacing, +rvir[0] * spacing)
    plt.ylim(-rvir[0] * spacing, +rvir[0] * spacing)

    plt.xlabel(r"$X\,(h^{-1}{\rm Mpc})$")
    plt.ylabel(r"$Y\,(h^{-1}{\rm Mpc})$")


def main():
    palette.configure(False)

    all_ids, mvir, rvir, mpeak, x, y, z = np.loadtxt(
        "halo_data.txt", usecols=(0, 1, 2, 3, 6, 7, 8)
    ).T
    vec = np.array([x, y, z]).T
    unique_ids = np.unique(all_ids)   # array of booleans for uniqueness
                                      # i.e. a new host halo id == True

    for j in range(5):
        plt.figure(j)
        ok = all_ids == unique_ids[j]   # SUPER HELPFUL LINE FOR CONDENSING
        plot_subhaloes(mpeak[ok], 1e12, rvir[ok], vec[ok])

    plt.show()
    """
    Ah, sorry, misread your question, it's an array of booleans. 
    So, if all_ids = np.array([1, 1, 2, 1, 3, 3]) and 
    unique_ids[j] = 1 , ok = np.array([True, True, False, True, False False])
    So when you index an array with ok , only the elements where ok[i] == True  are carried over.
    
    It may be helpful to play around with a test array in the Python shell or a new notebook.
    E.g. make some array x=np.arange(-10, 11) , and try using filtering to just print out 
    the positive numbers, the negative numbers, all the odd numbers, every number by -4, etc. 
    Try printing out the filtering array as you do this, and it'll help you get a better 
    intuition for what's going on.
    """


if __name__ == "__main__":
    main()