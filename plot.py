import numpy as np
import matplotlib.pyplot as plt
from utilities.plot_suiteSIAM import Plotter

plt.close("all")

pref = "./results/vvRF_"
er_fname = "errors_test_r_b.npy"
idx = 3

J_list = [256, 1024, 4096]
M_list = np.round(10**np.linspace(1, 4, 7)).astype(int)
N_list = np.round(10**np.linspace(1, np.log10(1548), 7)).astype(int)

# Fix N
N_fix = N_list[-1]
errors_M_sweep = np.zeros((3,7,2))
for i, J in enumerate(J_list):
    for j, M in enumerate(M_list):
        errors_M_sweep[i,j,...] = np.load(pref + 'idx' + str(idx)
                                          + '_J' + str(J)
                                          + 'm' + str(M)
                                          + 'n' + str(N_fix)
                                          + '/' + er_fname)

plt.figure(0)
plt.loglog(M_list, errors_M_sweep[...,0].T, 'o:')
plt.loglog(M_list, 1./M_list**.5)

plt.figure(1)
plt.loglog(M_list, errors_M_sweep[...,1].T, 'o:')
plt.loglog(M_list, 1./M_list**.5)

# Fix M
M_fix = M_list[-1]
errors_N_sweep = np.zeros((3,7,2))
for i, J in enumerate(J_list):
    for j, N in enumerate(N_list):
        errors_N_sweep[i,j,...] = np.load(pref + 'idx' + str(idx)
                                          + '_J' + str(J)
                                          + 'm' + str(M_fix)
                                          + 'n' + str(N)
                                          + '/' + er_fname)

plt.figure(10)
plt.loglog(N_list, errors_N_sweep[...,0].T, 'o:')
plt.loglog(N_list, 1./N_list**.25)

plt.figure(11)
plt.loglog(N_list, errors_N_sweep[...,1].T, 'o:')
plt.loglog(N_list, 1./N_list**.25)

plt.show()