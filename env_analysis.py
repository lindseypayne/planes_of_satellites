import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from scipy import stats
import numpy.random as random
from env_stats import get_axes
from theta_stats import opening_angle
from env_statistics import random_ellipsoid
from rvir_stats import new_tensor

"""
4/4/21
READ PAPER!!!
ks stat not powerful enough for VSMDPL 
E has higher resolution than V (resolves very small subhalos)
multi dark: 400, 160, 64
time checks/benchmarking & code efficiency thetas WHILE resolution tests, run on E and V, run jobs on Sherlock, ....
    TIME
        get a bunch of times for functions AND printing/table
            only do analysis on 1/4 hosts or every other on 160 
            start w/ thetas then sats
        go into longest time function and split up conceptual parts with time.time (use global variables)
        L63 is fastest, 64, 160, 400
    RES TESTS
        ca_rvir, f_corot, N_sats vs. Mass cutoff (this is my bin plot)
        minimum mass is 50x particle mass, maximum mass to 3xe11
        why?? trying to figure out where simulations overlap, where there are not numerical problems
            it's  possible shape, velocity, etc. of objects change based on resolution or group-type of simulation,
            some types of sims might be more accurate for certain measurements
        my question: the over-merging problem
            can argue using a model that the group vs. the one is returning more correct data
            we know model can predict and correct for simulation biases
"""
def main():

    L63_theta = np.load("L63theta_stats.npy")
    L63_sat = np.load("L63sat_stats.npy")
    L63_env = np.load("L63env_stats.npy")
    L63_rvir = np.load("L63rvir_stats.npy")
    L63p1, L63p2, L63thetas, L63rot = L63_theta[:, 1], L63_theta[:, 2], L63_theta[:, 3], L63_theta[:, 4]
    L63sat_host_idx, L63sub_idx, L63dx, L63dy, L63dz, L63sub_mvir, L63vx, L63vy, L63vz, L63_Lx, L63_Ly, L63_Lz = \
        np.array(L63_sat[:, 0], dtype=int), np.array(L63_sat[:, 1], dtype=int), L63_sat[:, 2], \
        L63_sat[:, 3], L63_sat[:, 4], L63_sat[:, 5], L63_sat[:, 6], L63_sat[:, 7], \
        L63_sat[:, 8], L63_sat[:, 9], L63_sat[:, 10], L63_sat[:, 11]
    L63host_idx, L63x, L63y, L63z, L63_Rvir, L63_Mvir, L63_ca_8, L63_ba_8, L63_ca_med, \
    L63_ba_med, L63_ddim, L63dbright, L63_xmin, L63_ymin, L63_zmin, L63_xmaj, L63_ymaj, \
    L63_zmaj = np.array(L63_env[:,0], dtype=int), L63_env[:,1], L63_env[:,2], L63_env[:,3],\
               L63_env[:, 4], L63_env[:,5], L63_env[:,6], L63_env[:,7], L63_env[:,8], L63_env[:,9], \
               L63_env[:,10], L63_env[:, 11], L63_env[:,12], L63_env[:,13], L63_env[:,14], \
               L63_env[:,15], L63_env[:,16], L63_env[:,17]
    L63host_idx, L63_N_sats, L63_ca_rvir, L63_ba_rvir, L63_host_rot = np.array(L63_rvir[:,0], dtype=int), \
                                               L63_rvir[:,1], L63_rvir[:,2], L63_rvir[:,3],L63_rvir[:, 4]

    ESMDPL_theta = np.load("ESMDPLtheta_stats.npy")
    ESMDPL_sat = np.load("ESMDPLsat_stats.npy")
    ESMDPL_env = np.load("ESMDPLenv_stats.npy")
    ESMDPL_rvir = np.load("ESMDPLrvir_stats.npy")
    Ep1, Ep2, Ethetas, Erot = ESMDPL_theta[:,1], ESMDPL_theta[:,2], ESMDPL_theta[:,3], ESMDPL_theta[:,4]
    Esat_host_idx, Esub_idx, Edx, Edy, Edz, Esub_mvir, Evx, Evy, Evz, E_Lx, E_Ly, E_Lz = \
        np.array(ESMDPL_sat[:,0], dtype=int), np.array(ESMDPL_sat[:,1], dtype=int), ESMDPL_sat[:,2], \
        ESMDPL_sat[:,3], ESMDPL_sat[:,4], ESMDPL_sat[:,5], ESMDPL_sat[:,6], ESMDPL_sat[:,7], \
        ESMDPL_sat[:,8], ESMDPL_sat[:,9], ESMDPL_sat[:,10], ESMDPL_sat[:,11]
    Ehost_idx, Ex, Ey, Ez, E_Rvir, E_Mvir, E_ca_8, E_ba_8, E_ca_med, \
    E_ba_med, E_ddim, Edbright, E_xmin, E_ymin, E_zmin, E_xmaj, E_ymaj, \
    E_zmaj = np.array(ESMDPL_env[:,0], dtype=int), ESMDPL_env[:,1], ESMDPL_env[:,2], ESMDPL_env[:,3],\
             ESMDPL_env[:, 4], ESMDPL_env[:,5], ESMDPL_env[:,6], ESMDPL_env[:,7], ESMDPL_env[:,8], \
             ESMDPL_env[:,9], ESMDPL_env[:,10], ESMDPL_env[:, 11], ESMDPL_env[:,12], ESMDPL_env[:,13], \
             ESMDPL_env[:,14], ESMDPL_env[:,15], ESMDPL_env[:,16], ESMDPL_env[:,17]
    Ehost_idx, E_N_sats, E_ca_rvir, E_ba_rvir, E_host_rot = np.array(ESMDPL_rvir[:, 0], dtype=int), \
                       ESMDPL_rvir[:, 1], ESMDPL_rvir[:, 2], ESMDPL_rvir[:,3], ESMDPL_rvir[:, 4]

    VSMDPL_theta = np.load("VSMDPLtheta_stats.npy")
    VSMDPL_sat = np.load("VSMDPLsat_stats.npy")
    VSMDPL_env = np.load("VSMDPLenv_stats.npy")
    VSMDPL_rvir = np.load("VSMDPLrvir_stats.npy")
    Vp1, Vp2, Vthetas, Vrot = VSMDPL_theta[:, 1], VSMDPL_theta[:, 2], VSMDPL_theta[:, 3], VSMDPL_theta[:, 4]
    Vsat_host_idx, Vsub_idx, Vdx, Vdy, Vdz, Vsub_mvir, Vvx, Vvy, Vvz, V_Lx, V_Ly, V_Lz = \
        np.array(VSMDPL_sat[:, 0], dtype=int), np.array(VSMDPL_sat[:, 1], dtype=int), VSMDPL_sat[:, 2], \
        VSMDPL_sat[:, 3], VSMDPL_sat[:, 4], VSMDPL_sat[:, 5], VSMDPL_sat[:, 6], VSMDPL_sat[:, 7], \
        VSMDPL_sat[:, 8], VSMDPL_sat[:, 9], VSMDPL_sat[:, 10], VSMDPL_sat[:, 11]
    Vhost_idx, Vx, Vy, Vz, V_Rvir, V_Mvir, V_ca_8, V_ba_8, V_ca_med, \
    V_ba_med, V_ddim,V_dbright,V_xmin, V_ymin, V_zmin, V_xmaj, V_ymaj, \
    V_zmaj = np.array(VSMDPL_env[:, 0], dtype=int), VSMDPL_env[:, 1], VSMDPL_env[:, 2], VSMDPL_env[:, 3], \
             VSMDPL_env[:, 4], VSMDPL_env[:, 5], VSMDPL_env[:, 6], VSMDPL_env[:, 7], VSMDPL_env[:, 8], \
             VSMDPL_env[:, 9], VSMDPL_env[:, 10], VSMDPL_env[:, 11], VSMDPL_env[:, 12], VSMDPL_env[:, 13], \
             VSMDPL_env[:, 14], VSMDPL_env[:, 15], VSMDPL_env[:, 16], VSMDPL_env[:, 17]
    Vhost_idx, V_N_sats, V_ca_rvir, V_ba_rvir, V_host_rot = np.array(VSMDPL_rvir[:, 0], dtype=int), \
                        VSMDPL_rvir[:, 1], VSMDPL_rvir[:, 2], VSMDPL_rvir[:,3], VSMDPL_rvir[:, 4]

    SMDPL_theta = np.load("SMDPLtheta_stats.npy")
    SMDPL_sat = np.load("SMDPLsat_stats.npy")
    SMDPL_env = np.load("SMDPLenv_stats.npy")
    SMDPL_rvir = np.load("SMDPLrvir_stats.npy")
    Sp1, Sp2, Sthetas, Srot = SMDPL_theta[:, 1], SMDPL_theta[:, 2], SMDPL_theta[:, 3], SMDPL_theta[:, 4]
    Ssat_host_idx, Ssub_idx, Sdx, Sdy, Sdz, Ssub_mvir, Svx, Svy, Svz, S_Lx, S_Ly, S_Lz = \
        np.array(SMDPL_sat[:, 0], dtype=int), np.array(SMDPL_sat[:, 1], dtype=int), SMDPL_sat[:, 2], \
        SMDPL_sat[:, 3], SMDPL_sat[:, 4], SMDPL_sat[:, 5], SMDPL_sat[:, 6], SMDPL_sat[:, 7], \
        SMDPL_sat[:, 8], SMDPL_sat[:, 9], SMDPL_sat[:, 10], SMDPL_sat[:, 11]
    Shost_idx, Sx, Sy, Sz, S_Rvir, S_Mvir, S_ca_8, S_ba_8, S_ca_med, \
    S_ba_med, S_ddim, S_dbright, S_xmin, S_ymin, S_zmin, S_xmaj, S_ymaj, \
    S_zmaj = np.array(SMDPL_env[:, 0], dtype=int), SMDPL_env[:, 1], SMDPL_env[:, 2], SMDPL_env[:, 3], \
             SMDPL_env[:, 4], SMDPL_env[:, 5], SMDPL_env[:, 6], SMDPL_env[:, 7], SMDPL_env[:, 8], \
             SMDPL_env[:, 9], SMDPL_env[:, 10], SMDPL_env[:, 11], SMDPL_env[:, 12], SMDPL_env[:, 13], \
             SMDPL_env[:, 14], SMDPL_env[:, 15], SMDPL_env[:, 16], SMDPL_env[:, 17]
    Shost_idx, S_N_sats, S_ca_rvir, S_ba_rvir, S_host_rot = np.array(SMDPL_rvir[:, 0], dtype=int), \
                                                            SMDPL_rvir[:, 1], SMDPL_rvir[:, 2], SMDPL_rvir[:,
                                                                                                  3], SMDPL_rvir[:, 4]

    ids = [[L63host_idx, L63sat_host_idx], [Ehost_idx, Esat_host_idx], [Vhost_idx, Vsat_host_idx], [Shost_idx, Ssat_host_idx]]
    ds = [[L63dx, L63dy, L63dz], [Edx, Edy, Edz], [Vdx, Vdy, Vdz], [Sdx, Sdy, Sdz]]
    vs = [[L63vx, L63vy, L63vz], [Evx, Evy, Evz], [Vvx, Vvy, Vvz], [Svx, Svy, Svz]]
    mvirs = [L63sub_mvir, Esub_mvir, Vsub_mvir, Ssub_mvir]
    m_pairs = [[L63p1, L63p2], [Ep1, Ep2], [Vp1, Vp2], [Sp1, Sp2]]
    thetas = [L63thetas, Ethetas, Vthetas, Sthetas]
    rots = [L63rot, Erot, Vrot, Srot]

    #convergence([L63host_idx, L63sat_host_idx], [L63dx, L63dy, L63dz], [L63vx, L63vy, L63vz], L63sub_mvir, [L63p1, L63p2], L63thetas, L63rot)
    theta_mass([L63p1, L63p2], L63thetas, L63rot)
    new_rot_convergence([L63p1, L63p2], L63thetas, L63rot)
    convergence_all(ids, ds, vs, mvirs, m_pairs, thetas, rots)

    """euc_dist = []
    sim_labels = []
    env_props = [ca_8, ba_8, ddim, ddim/dbright]
    all_props = [ca_med, ba_med, ca_rvir, ba_rvir, ddim, ddim/dbright]
    env_labels = [r'$c/a_{8}$', r'$b/a_{8}$', r'$\Delta_{dim}$', r'$\Delta_{ratio}$']
    all_labels = ['ca_med', 'ba_med', 'ca_rvir', 'ba_rvir', 'ddim', 'dratio']
    for i in range(4):
        props, MW_data, labels = user_choice(ca_med, ba_med, ddim, ddim/dbright)
        euc = euc_distance(props, all_props, MW_data, labels, all_labels, ca_rvir, ba_rvir, corotation)
        euc_dist.append(np.asarray(euc))
        sim_labels.append(labels)
    theta_env_MW([host_idx, sat_host_idx], [np.asarray(euc_dist), sim_labels], [env_props, env_labels],
                 thetas, rot, 'MW') # percentiles plotted for thetas
    theta_env_MW([host_idx, sat_host_idx], [np.asarray(euc_dist), sim_labels], [env_props, env_labels],
                 thetas, rot, 'env') # percentiles plotted for thetas
    theta_mass(host_idx,[mvir_sat1, mvir_sat2], thetas, rot)
 
    env_percentiles(corotation, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir)
                # percentiles plotted for axis ratios
    props, MW_data, labels = user_choice(ca_med, ba_med, ddim, ddim / dbright)
    euc_distance(props, all_props, MW_data, labels, all_labels, ca_rvir, ba_rvir, corotation)
                # percentiles plotted for axis ratios

    test_points()
    test_ca_recovery()"""


def get_pvalue(full_sample, subsample, N_loops, bool):
    """Computes the statistical probability value of
    a selected sample of halos being drawn from the
    full distribution of halos in the simulation using
    either the Kolmogorov Smirnov or Anderson Darling test.
    """
    if bool:
        stat, _ = stats.kstest(subsample, full_sample)
    elif not bool:
        stat, _, _ = stats.anderson_ksamp([subsample, full_sample])
    count = 0
    for i in range(N_loops):
        num_points = len(subsample)
        ran_sample = np.random.choice(full_sample, num_points, replace=True)
        if bool:
            stat_emp, _ = stats.kstest(ran_sample, full_sample)
        elif not bool:
            stat_emp, _, _ = stats.anderson_ksamp([ran_sample, full_sample])
        if stat_emp > stat:
            count += 1
    updated_pval = count / N_loops
    return updated_pval


def histenv(vara, varb, ax, per1, labela, labelb, setax, upperx,
            uppery, bins, subtitle, colors, xtitle, ytitle):
    sm_ratio_lim = vara < np.percentile(vara, per1)
    # vara: env, varb: sat
    N_loops = 1000
    KS_pval = get_pvalue(varb, varb[sm_ratio_lim], N_loops, True)
    AD_pval = get_pvalue(varb, varb[sm_ratio_lim], N_loops, False)
    # each of these curves are CDFs
    ax.hist(varb[sm_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0, upperx), color=colors[1], label='KS pvalue ' + str(round(KS_pval, 2)))
    ax.hist(varb[sm_ratio_lim], bins=bins, cumulative=True, density=True, histtype='step',
            range=(0, upperx), color=colors[1], label='AD pvalue ' + str(round(AD_pval, 2)))
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
def env_percentiles(corotations, ca_8, ba_8, ddim, dbright, ca_rvir, ba_rvir):
    """ Creating another 18 plots that splits only on environmental variable and plots satellite
    variable properties on the x and y axes. First row of plots splits on the top and bottom 5th
    percentile, then 10th percentile, then 25th.

    First plot shows scatter distribution. Second shows histogram the statistical probability
    (at different percentage splits on the population) of a sample population belonging to
    the same distribution as the parent population. Null hypothesis is that they do.
    """
    dratio=ddim/dbright
    fig, ax = plt.subplots(3, 6, figsize=(14, 14), dpi=100, tight_layout=True)
    i = 0
    upperx = 1.0
    uppery = 1.0
    bins = 200
    per_list = [1, 3, 5]
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
        histenv(ca_8, ca_rvir, ax[i][0], per, r'$(c/a)_{8Mpc}$',
                r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, True)
        histenv(ca_8, corotations, ax[i][1], per, r'$(c/a)_{8Mpc}$',
                r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(ddim, ca_rvir, ax[i][2], per, r'$\Delta_{dim}$',
                r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(ddim, corotations, ax[i][3], per, r'$\Delta_{dim}$',
                r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(dratio, ca_rvir, ax[i][4], per, r'$\Delta_{ratio}$',
                r'$(c/a)_{rvir}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        histenv(dratio, corotations, ax[i][5], per, r'$\Delta_{ratio}$',
                r'$f_{corotation}$', setax, upperx, uppery, bins, title, colors[i], xtitle, ytitle)
        i += 1
    fig.suptitle('Histogram of Satellite Properties Split on Environmental '
                 'Property Percentiles',
                 y=0.97)
    line2 = mlines.Line2D([], [], color='deepskyblue', label='1%')
    line4 = mlines.Line2D([], [], color='purple', label='3%')
    line6 = mlines.Line2D([], [], color='darkturquoise', label='5%')
    line7 = mlines.Line2D([], [], color='black', label='all hosts')
    fig.legend(handles=[line2, line4, line6, line7], fontsize=6)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    plt.show()


def user_choice(ca_med, ba_med, ddim, dratio):
    """Asks user for inputted host properties to make a list of variables
    for the argument of sat_MW_comparison.
    """
    props = [ca_med, ba_med, ddim, dratio]
    # from table 3 Neuzil et al 2019
    MW = [0.163, 0.786, 1.699, 1.699/5.190]
    labels = [r'$(c/a)_{median}$ ', r'$(b/a)_{median}$ ',
              r'$\Delta_{dim}$ ', r'$\Delta_{ratio}$']
    propsa = []
    MWa = []
    labelsa = ''
    a = str(input("Choose properties by entering a number 0-3 corresponding to "
                  "ca_med, ba_med, ddim, dratio. Press SPACE to end. "))
    while a != "":
        a = int(a)
        propsa.append(props[a])
        MWa.append(MW[a])
        labelsa += labels[a]
        a = str(input("Choose properties by entering a number 0-3 corresponding to "
                      "ca_med, ba_med, ddim, dratio. Press RETURN to end: "))
    return propsa, MWa, labelsa


def histMW(vara, varb, ax, per, labelb, setax, upperx,            # vara: env, varb: sat
               uppery, bins, colors, xtitle, ytitle, leg):
    lower_lim = vara < np.percentile(vara, per)
    N_loops = 1000  # change to 50k
    KS_pval = get_pvalue(varb, varb[lower_lim], N_loops, True)
    AD_pval = get_pvalue(varb, varb[lower_lim], N_loops, False)
    # each of these curves are CDFs
    ax.hist(varb[lower_lim], bins=bins, cumulative=True, density=True, range=(0, upperx),
            histtype='step', color=colors[1], label='KS pvalue ' + str(round(KS_pval,2)))
    ax.hist(varb[lower_lim], bins=bins, cumulative=True, density=True, range=(0, upperx),
            histtype='step', color=colors[1], label='AD pvalue ' + str(round(AD_pval,2)))
    ax.hist(varb, bins=bins, cumulative=True, density=True, range=(0, upperx),
            histtype='step', color=colors[2])
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


def MWsimilarity_percentiles(sat1, sat2, euc_dist, label1, label2, sim_components):
    fig, ax = plt.subplots(3, 2, figsize=(10, 10), dpi=75, tight_layout=False)
    upperx = 1.0
    uppery = 1.0
    bins = 200
    per_list = [1, 3, 5]
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
    line1 = mlines.Line2D([], [], color='indianred', label='1% most similar')
    line3 = mlines.Line2D([], [], color='forestgreen', label='3% most similar')
    line5 = mlines.Line2D([], [], color='purple', label='5% most similar')
    line7 = mlines.Line2D([], [], color='black', label='all hosts')
    comment = mpatches.Patch(edgecolor='black', facecolor='white', label='AD: Anderson-Darling Test')
    comment2 = mpatches.Patch(edgecolor='black', facecolor='white', label='KS: Kolmogorov-Smyrnov Test')
    comment3 = mpatches.Patch(edgecolor='black', facecolor='white', label='similarity computed with: '+ sim_components)
    fig.legend(handles=[line1, line3, line5, line7, comment, comment2, comment3], fontsize=8)
    plt.subplots_adjust(wspace=0.2)
    plt.show()


# two of two ways to select subsamples
def euc_distance(props, all_props, MW_data, labels, all_labels, ca_rvir, ba_rvir, corotations):
    """This function ranks each MW analog on its similarity to the
    Milky Way using a euclidean distance function. Returns an array
    with a "similarity distance" for each halo.
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
    MWsimilarity_percentiles(corotations, ca_rvir, euc_dist, label1=r'$f_{corotation}$',
                       label2=r'$(c/a)_{rvir}$', sim_components=labels)
    MWsimilarity_percentiles(corotations, ba_rvir, euc_dist, label1=r'$f_{corotation}$',
                       label2=r'$(b/a)_{rvir}$', sim_components=labels)
    most_indices = np.argpartition(euc_dist,range(5))[:5]
    list_2d = []
    i = 0
    for index in most_indices:
        list = []
        for prop in all_props:
            list.append(prop[index])
        list_2d.append(list)
        i += 1
    return euc_dist


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


def theta_xy(angles):
    centers = []
    pdens = []
    bins = 10
    for a in angles:
        N, theta_edges = np.histogram(a, range=(0, np.pi), bins=bins)
        bin_size = np.pi / bins
        p_density = N / np.sum(N) / bin_size
        theta_centers = (theta_edges[1:] + theta_edges[:-1]) / 2
        sin_theta = np.sin(theta_centers) / 2
        scaled_pdensity = p_density / sin_theta
        centers.append(theta_centers)
        pdens.append(scaled_pdensity)
    return centers, pdens


def theta_xyROT(co, ant):
    bins = 10

    Nc, theta_edges_c = np.histogram(co, range=(0, np.pi), bins=bins)
    Na, theta_edges_a = np.histogram(ant, range=(0, np.pi), bins=bins)
    bin_size = np.pi / bins
    density_c = Nc / (np.sum(Nc) + np.sum(Na)) / bin_size
    density_a = Na / (np.sum(Nc) + np.sum(Na)) / bin_size
    theta_centers_c = (theta_edges_c[1:] + theta_edges_c[:-1]) / 2
    theta_centers_a = (theta_edges_a[1:] + theta_edges_a[:-1]) / 2
    sin_theta_c = np.sin(theta_centers_c) / 2
    sin_theta_a = np.sin(theta_centers_a) / 2
    scaled_density_c = density_c / sin_theta_c
    scaled_density_a = density_a / sin_theta_a

    ratio_dens = scaled_density_c / scaled_density_a
    return ratio_dens


# HAS TO BE LONG
def theta_and_convergence_plots(selection, ax, limit, ax_title, ids, dict, thetas, rot, kind):
    if kind == 'env' or kind == 'MW':
        sample_bool = selection[0] < np.percentile(selection[0], limit) # hosts below percentile, bool array
        sample_host_ids = ids[0][sample_bool]  # idx array of hosts
        sample_list = []
        for id in sample_host_ids:
            sample_list.append(np.where(ids[1] == id)[0])
        sample = np.hstack(sample_list)  # idx array for every theta pair of host below percentil
    if kind == 'ca':
        # id in target_ids[0]
        ca = []
        N_subs = []
        rots = []
        for h in ids[0]:
            # sub masses of this host
            id_match = np.where(ids[1] == h)[0]
            mvir_subs = selection[0][id_match]
            dx = selection[1][0][id_match]
            dy = selection[1][1][id_match]
            dz = selection[1][2][id_match]
            vx = selection[2][0][id_match]
            vy = selection[2][1][id_match]
            vz = selection[2][2][id_match]
            above = mvir_subs > limit
            N_sub = len(mvir_subs[above])
            if len(dx[above]) >=4:
                rvir = get_axes(dx[above], dy[above], dz[above])
                ca.append(rvir[0][0])
                corot, _ = new_tensor(dx[above], dy[above], dz[above], vx[above], vy[above], vz[above])
                rots.append(corot)
            else:
                ca.append(np.nan)
                rots.append(np.nan)
            N_subs.append(N_sub)
        return rots, N_subs, ca
    if kind == 'mass' or kind == 'convergence' or 'rot convergence':
        id = 0
        sample_list = []
        for z in range(len(selection[0])):
            if selection[0][z] > limit and selection[1][z] > limit:
                sample_list.append(id)
            id += 1
        sample = np.asarray(sample_list)   # idx array for every theta pair of above mass limit
    all_thetas = thetas
    select_thetas = thetas[sample]
    corot = thetas[np.where(rot > 0)[0]]
    anti = thetas[np.where(rot < 0)[0]]
    #select_corot = thetas[np.where(rot[sample] > 0)[0]]
    #select_anti = thetas[np.where(rot[sample] < 0)[0]]
    """
    2 plots that didn't make physical sense/unconsistent: fcorot vs. Mmin to pair-based fcorot
    change Mmin cutoffs and examine smallest mass pair values in sim, notice unreasonable jump
    step through function: does each make sense?
    FIND BUG!
    """
    select_corot = select_thetas[rot[sample] > 0]
    select_anti = select_thetas[rot[sample] < 0]
    #print(len(all_thetas), len(corot), len(anti), len(select_thetas), len(select_corot), len(select_anti))
    angles = [all_thetas, corot, anti, select_thetas, select_corot, select_anti]
    centers, pdens = theta_xy(angles)
    if kind == 'convergence':
        return centers[3], pdens[3]
    elif kind == 'rot convergence':
        select_thetas_y = pdens[3]
        #center_ratio = centers[4] / centers[5]
        #pdens_ratio = pdens[4] / pdens[5]
        #return centers[3], pdens[3], center_ratio, pdens_ratio
        select_ratio_y = theta_xyROT(select_corot, select_anti)
        select_corot_y = pdens[4]
        select_anti_y = pdens[5]
        return select_thetas_y, select_corot_y, select_anti_y, select_ratio_y
    ax.axhline(y=1, color='black', lw=0.8, linestyle='--')
    ax.plot(centers[0], pdens[0], color='blue')
    ax.plot(centers[1], pdens[1], color='red')
    ax.plot(centers[2], pdens[2], color='purple')
    if kind == 'env' or kind == 'MW':
        host_ids = np.arange(len(ids[0]))
        ps, _ = lilliefors(host_ids, all_thetas, select_thetas, N_loops=10)
        pc, _ = lilliefors(host_ids, corot, select_corot, N_loops=10)
        pa, _ = lilliefors(host_ids, anti, select_anti, N_loops=10)
        ax.plot(centers[3], pdens[3], label=r'$p$ = ' + str(round(ps, 2)), color='blue', ls='--')
        ax.plot(centers[4], pdens[4], label=r'$p$ = ' + str(round(pc, 2)), color='red', ls=':')
        ax.plot(centers[5], pdens[5], label=r'$p$ = ' + str(round(pa, 2)), color='purple', ls='-.')
        string = str(limit) + 'percentile'
        dict[string] = {r'$p$-values': {'all-sample': ps, 'corot-sample': pc, 'antirot-sample': pa}}
        if kind == 'env':
            ax.set_title(str(limit) + 'th percentile ' + str(selection[1]), fontsize=10)
        else:
            ax.set_title(str(limit) + 'th percentile similarity: ' + str(selection[1]), fontsize=10)
    elif kind == 'mass':
        ax.plot(centers[3], pdens[3], color='blue', ls='--')
        ax.plot(centers[4], pdens[4], color='red', ls=':')
        ax.plot(centers[5], pdens[5], color='purple', ls='-.')
        ax.set_title(r'$M_{sat} >$ ' + f'{limit:.1e}')
    if ax_title[0] is True:
        ax.set_xlabel(r'Opening angle $\theta_{open}$ [rad]', fontsize=10)
    if ax_title[1] is True:
        ax.set_ylabel(r'$N_{\theta_{open}}\ /\ N_{tot}\ /\ d(\theta_{open})/\ sin(\theta)$', fontsize=10)
    ax.set_ylim(0.8, 1.6)
    ax.set_aspect(4)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="best", fontsize=8)
    return dict


def theta_env_MW(hosts, MW, env, thetas, rot, kind):
    fig, ax = plt.subplots(3,4,figsize=(10, 10), dpi=75, tight_layout=False)
    i = 0
    bins=10
    per_list = [5, 10, 25]
    ytitle = False
    xtitle = False
    dict = {}
    if kind == 'env':
        fig.suptitle('CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_'
                     'L63 Split on Single Environmental Parameters')
        selection_var = [env[0][0], env[0][1], env[0][2], env[0][3]]
        selection_label = [env[1][0], env[1][1], env[1][2], env[1][3]]
    elif kind == 'MW':
        fig.suptitle('CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_'
                     'L63 selected on Percentiles of Similarity to MW')
        selection_var = [MW[0][0], MW[0][1], MW[0][2], MW[0][3]]
        selection_label = [MW[1][0], MW[1][1], MW[1][2], MW[1][3]]
    for per in per_list:
        if i == 2:
            xtitle = True
        theta_and_convergence_plots([selection_var[0], selection_label[0]], ax[i][0], per, [xtitle, True],
                    hosts, dict, thetas, rot, kind)
        theta_and_convergence_plots([selection_var[1], selection_label[1]], ax[i][1], per, [xtitle, ytitle],
                    hosts, dict, thetas, rot, kind)
        theta_and_convergence_plots([selection_var[2], selection_label[2]], ax[i][2], per, [xtitle, ytitle],
                    hosts, dict, thetas, rot, kind)
        theta_and_convergence_plots([selection_var[3], selection_label[3]], ax[i][3], per, [xtitle, ytitle],
                    hosts, dict, thetas, rot, kind)
        i += 1
    print(dict)
    line1 = mlines.Line2D([], [], label=r'all $\theta_{open}$ pairs', color='blue')
    line2 = mlines.Line2D([], [], label='all corotating pairs', color='red')
    line3 = mlines.Line2D([], [], label='all antirotating pairs', color='purple')
    line4 = mlines.Line2D([], [], color='black', lw=0.8, linestyle='--', label='expected spherical dist')
    line5 = mlines.Line2D([], [], lw=0.8, label='top percentile all pairs', color='blue', ls='--')
    line6 = mlines.Line2D([], [], lw=0.8, label='top percentile corot pairs', color='red', ls=':')
    line7 = mlines.Line2D([], [], lw=0.8, label='top percentile antirot pairs', color='purple', ls='-.')
    fig.legend(handles=[line1, line2, line3, line4, line5, line6, line7], fontsize=10)
    plt.show()


##### differences????
def theta_mass(mvir_pairs, thetas, rot):
    fig, axs = plt.subplots(1, 3, figsize=(10, 10), dpi=75, tight_layout=False)
    i = 0
    lower_limits = [1e9, 1e10, 1e11]
    ytitle = True
    xtitle = True
    dict = {}
    for ax in axs:
        print()
        print("i", i)
        if i == 1 or i == 2:
            ytitle = False
        if i == 2:
            print(np.sum(mvir_pairs[1] < lower_limits[i]))
        theta_and_convergence_plots([mvir_pairs[0], mvir_pairs[1]], ax, lower_limits[i], [xtitle, ytitle],
                    [None, None], dict, thetas, rot, 'mass')
        i += 1
    fig.suptitle('CDFs of Angular Separation Between Satellite Pairs in Erebos_CBol_L63 '
                 'selected on Subhalo Mass Limits')
    line1 = mlines.Line2D([], [], label=r'all $\theta_{open}$ pairs', color='blue')
    line2 = mlines.Line2D([], [], label='all corotating pairs', color='red')
    line3 = mlines.Line2D([], [], label='all antirotating pairs', color='purple')
    line4 = mlines.Line2D([], [], color='black', lw=0.8, linestyle='--', label='expected spherical dist')
    line5 = mlines.Line2D([], [], lw=0.8, label='all halos > mass limit', color='blue', ls='--')
    line6 = mlines.Line2D([], [], lw=0.8, label='corot > mass limit', color='red', ls=':')
    line7 = mlines.Line2D([], [], lw=0.8, label='antirot > mass limit', color='purple', ls='-.')
    fig.legend(handles=[line1, line2, line3, line4, line5, line6, line7], fontsize=10)
    plt.show()


L63_M_min = 1.7e7
def new_rot_convergence(mvir_pairs, thetas, rot):
    limits = np.geomspace(50 * L63_M_min, 3e11, num=25, endpoint=True)
    c = ["blue", "red", "green", "orange"]
    bin_choice = 0  # 0-9
    y_all = []
    y_co = []
    y_ant = []
    y_ratios = []
    print(rot)    ###### this is wrong!!!!!!!
    for l in range(len(limits)):
        all_y, co_y, ant_y, ratio_y = theta_and_convergence_plots([mvir_pairs[0], mvir_pairs[1]], None, limits[l],
                                [None, None], None, None, thetas, rot, 'rot convergence')
        y_all.append(all_y[bin_choice])
        y_co.append(co_y[bin_choice])
        y_ant.append(ant_y[bin_choice])
        y_ratios.append(ratio_y[bin_choice])
    y_label = r'???'
    fig, ax = plt.subplots(dpi=100)
    y = [y_all, y_co, y_ant, y_ratios]
    ax.plot(limits, y[0], color=c[0], label=r'all > $M_{min}$')
    ax.plot(limits, y[1], color=c[1], label=r'$f_{corot}$')
    ax.plot(limits, y[2], color=c[2], label=r'$f_{antirot}$')
    ax.plot(limits, y[3], color=c[3], label=r'$f_{corot/antirot}$')
    ax.set_xscale("log")
    ax.set_xlabel(r'$M_{min}$')
    ax.set_ylabel(y_label)
    fig.suptitle(r'$f_{rot}$ Convergence Test for Milky Way Analogs in L63')
    ax.legend()
    plt.show()


def convergence(ids, disps, vels, mvirs, mvir_pairs, thetas, rot):
    limits = np.geomspace(50*L63_M_min, 3e11, num=25, endpoint=True)
    c = ["blue", "red", "green", "orange"]
    bin_choice = 0  # 0-9
    y_data_theta = []
    y_data_ca = []
    y_data_rot = []
    y_data_N = []
    for l in range(len(limits)):
        _, y0 = theta_and_convergence_plots([mvir_pairs[0], mvir_pairs[1]], None, limits[l], [None, None],
                           None, None, thetas, rot, 'convergence')
        y_data_theta.append(y0[bin_choice])
        y3, y2, y1 = theta_and_convergence_plots([mvirs, disps, vels], None, limits[l], [None, None],
                                                 ids, None, thetas,rot, kind='ca')
        y1 = np.array(y1)
        y2 = np.array(y2)
        y3 = np.array(y3)
        y_data_ca.append(np.mean(y1[~np.isnan(y1)]))
        y_data_N.append(np.mean(y2[~np.isnan(y2)]))
        y_data_rot.append(np.mean(y3[~np.isnan(y3)]))
    y = [y_data_theta, y_data_ca, y_data_N, y_data_rot]
    y_labels = [r'$N_{\theta_{open}}(\theta_{open})\ /\ N_{tot}\ /\ d(\theta_{open})/\ sin(\theta)$',
                r'$c/a_{rvir}$', r'$N_{sat}$ per host', r'$f_{corot}$']
    for p in range(4):
        fig, ax = plt.subplots(dpi=100)
        ax.plot(limits, y[p], color=c[p])
        ax.set_xscale("log")
        ax.set_xlabel(r'$M_{min}$')
        ax.set_ylabel(y_labels[p])
        if p == 2:
            ax.set_yscale('log')
        fig.suptitle('Convergence Tests for Milky Way Analogs in L63')
        line1 = mlines.Line2D([], [], lw=0.8, label=r'halos > $M_{min}$ L63', color=c[p])
        line2 = line2 = mlines.Line2D([], [], lw=0.8, color='white', ls='-')
        if p == 0:
            line2 = mlines.Line2D([], [], lw=0.8, label='bin choice: ' + str(bin_choice), color='white', ls='-')
        fig.legend(handles=[line1, line2], fontsize=10)
    plt.show()


def convergence_all(ids, disps, vels, mvirs, mvir_pairs, thetas, rot):
    M_min = [1.7e7, 2.6e6, 6.2e6, 9.6e7]
    c = ["blue", "red", "green", "orange"]
    bin_choice = 0  # 0-9
    y_data_theta = []
    y_data_ca = []
    y_data_rot = []
    y_data_N = []
    for x in range(4):
        sim_y_theta = []
        sim_y_ca = []
        sim_y_rot = []
        sim_y_N = []
        limits = np.geomspace(50 * M_min[x], 3e11, num=25, endpoint=True)
        for l in range(len(limits)):
            _, y0 = theta_and_convergence_plots([mvir_pairs[x][0], mvir_pairs[x][1]], None, limits[l], [None, None],
                                                None, None, thetas[x], rot[x], 'convergence')
            sim_y_theta.append(y0[bin_choice])
            y3, y2, y1 = theta_and_convergence_plots([mvirs[x], disps[x], vels[x]], None, limits[l], [None, None],
                                                     ids[x], None, thetas[x], rot[x], 'ca')
            y1 = np.array(y1)
            y2 = np.array(y2)
            y3 = np.array(y3)
            sim_y_ca.append(np.mean(y1[~np.isnan(y1)]))
            sim_y_N.append(np.mean(y2[~np.isnan(y2)]))
            sim_y_rot.append(np.mean(y3[~np.isnan(y3)]))
        y_data_theta.append(sim_y_theta)
        y_data_ca.append(sim_y_ca)
        y_data_rot.append(sim_y_rot)
        y_data_N.append(sim_y_N)
    y = [y_data_theta, y_data_ca, y_data_N, y_data_rot]
    y_labels = [r'$N_{\theta_{open}}(\theta_{open})\ /\ N_{tot}\ /\ d(\theta_{open})/\ sin(\theta)$',
                    r'$c/a_{rvir}$', r'$N_{sat}$ per host', r'$f_{corot}$']

    for p in range(4):
        fig, ax = plt.subplots(dpi=100)
        x0 = np.geomspace(50 * M_min[0], 3e11, num=25, endpoint=True)
        x1 = np.geomspace(50 * M_min[1], 3e11, num=25, endpoint=True)
        x2 = np.geomspace(50 * M_min[2], 3e11, num=25, endpoint=True)
        x3 = np.geomspace(50 * M_min[3], 3e11, num=25, endpoint=True)
        ax.plot(x0, y[p][0], color=c[0], label="L63")
        ax.plot(x1, y[p][1], color=c[1], label="ESMDPL")
        ax.plot(x2, y[p][2], color=c[2], label="VSMDPL")
        ax.plot(x3, y[p][3], color=c[3], label="SMDPL")
        ax.set_xscale("log")
        #ax.set_ylim(1, 3.5) #!!!!
        ax.set_xlabel(r'$M_{min}$')
        ax.set_ylabel(y_labels[p])
        if p == 2:
            ax.set_yscale('log')
        fig.suptitle('Convergence Tests for Milky Way Analogs in 4 DM Sims')
        if p == 0:
            line = mlines.Line2D([], [], lw=0.8, label='bin choice: ' + str(bin_choice), color='white', ls='-')
            fig.legend(handles=[line], fontsize=10)
        ax.legend()
    plt.show()



########################################################################################################################
# working with fake data distributions below


def random_ca(ax, a, b, c, j, shape):
    N_realizations = 1000
    N_points = [5, 10, 20, 40, 80]
    colors = ['red', 'green', 'blue', 'purple']
    ca_exp = c/a
    ca_2d = []
    ca_stds = []
    N_2d = []
    ca_mean = []
    for n in N_points:
        ca_point = []
        N_point = []
        for i in range(N_realizations):
            x, y, z = random_ellipsoid(n, a, b, c)
            ca, ba, _ = get_axes(x, y, z)
            ca_point.append(ca)
            N_point.append(n)
        ca_2d.append(ca_point)
        ca_stds.append(np.std(ca_point))
        ca_mean.append(np.mean(ca_point))
        N_2d.append(N_point)
    ax.axhline(y=ca_exp, color='black', lw=0.8, linestyle='--',
               label=r'$c/a_{expected}$ for ' + str(shape))
    ax.errorbar(N_points, ca_mean, yerr=ca_stds, color=colors[j],
                ecolor=colors[j], xerr=None, ls='none')
    ax.scatter(N_points, ca_mean, color=colors[j], marker='x',
               label=r'$c/a_{mean}$ and $c/a_{recovered}$')
    ax.set_xscale('log')
    ax.legend()
    ax.set_ylim(0,1.1)


def test_ca_recovery():
    fig, ax = plt.subplots(2,2, figsize=(10, 10), dpi=75)
    N_realizations = 1000
    random_ca(ax[0][0], 1, 1, 1, 0, 'sphere')   # sphere
    random_ca(ax[0][1], 1, 0.5, 0.5, 1, 'prolate ellipsoid')  # prolate, small axes similar
    random_ca(ax[1][0], 1, 1, 0.75, 2, 'oblate ellipsoid')   # oblate, big axes similar
    random_ca(ax[1][1], 1, 0.75, 0.75, 3, 'prolate ellipsoid') # prolate, small axes similar

    ax[1][0].set_xlabel(r'$N_{points}$')
    ax[1][1].set_xlabel(r'$N_{points}$')
    ax[0][0].set_ylabel(r'$(c/a_{recovered})\ / \ (c/a_{expected})$')
    ax[1][0].set_ylabel(r'$(c/a_{recovered})\ / \ (c/a_{expected})$')
    fig.suptitle(r'Axis Ratio Recovery for Various $N_{sats}$ Done With ' +
                 str(N_realizations) + ' Point Realizations')
    plt.show()


def test_thetas(n, a, b, c, ax, labelx, labely, bins, color):
    all_pdensities = []
    # using 1000 ellipses
    for r in range(1000):
        x, y, z = random_ellipsoid(n, a, b, c)
        idx = np.arange(len(x))
        i, j = idx, idx
        i_grid, j_grid = np.meshgrid(i, j)
        i_flat, j_flat = i_grid.flatten(), j_grid.flatten()
        valid_pair = j_flat > i_flat
        i_flat, j_flat = i_flat[valid_pair], j_flat[valid_pair]
        thetas = opening_angle(x[i_flat], y[i_flat], z[i_flat],
                               x[j_flat], y[j_flat], z[j_flat])
        N, theta_edges = np.histogram(thetas, range=(0, np.pi), bins=bins)
        bin_size = np.pi / bins
        p_density = N / np.sum(N) / bin_size
        theta_centers = (theta_edges[1:] + theta_edges[:-1]) / 2
        sin_theta = np.sin(theta_centers) / 2
        scaled_pdensity = p_density / sin_theta
        all_pdensities.append(scaled_pdensity)
    avg_pdensities = np.average(all_pdensities, axis=0)  #axis=0
    ax.plot(theta_centers, avg_pdensities, label='test points: ' + str(n), color=color)
    ax.set_xlabel(labelx, fontsize=10)
    ax.set_ylabel(labely, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="best")


def test_points():
    fig, ax = plt.subplots(figsize=(10, 10), dpi=75, tight_layout=False)
    a = 1
    b = 0.5
    c = 0.25
    bins=10
    points_list = [5, 10, 15, 20]
    colors = ['red', 'green', 'blue', 'purple']
    labelx = r'Opening angle $\theta_{open}$ [rad]'
    labely = r'$N_{\theta_{open}}\ /\ N_{tot}\ /\ d(\theta_{open})/\ sin(\theta)$'
    ax.axhline(y=1, color='white', lw=0.8, linestyle='',
               label='a,b,c lengths: ' + str(a) + ', ' + str(b) + ', '+ str(c))
    ax.axhline(y=1, color='black', lw=0.8, linestyle='--', label='expected spherical dist')
    for p in range(len(points_list)):
        test_thetas(points_list[p], a, b, c, ax, labelx, labely, bins, colors[p])
    fig.suptitle(r'CDFs of Angular Separation Between Satellite Pair Analogs '
                 r'(test points) in oblate ellipsoid selected on $N_{test points}$')
    ax.legend(loc="best")
    plt.show()


if __name__ == '__main__':
    main()