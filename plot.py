import numpy as np
import matplotlib.pyplot as plt
from utilities.plot_suiteSIAM import Plotter

plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rc('legend', fontsize=12)

plt.close("all")

# pref = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/results/vvRF_'
pref = "./results/vvRF_"
er_fname = "errors_test_r_b.npy"
idx_max = 9
# idx = 0
FLAG_SAVE = not True
n_std = 2

J_list = [256, 1024, 4096]
# M_list = np.round(10**np.linspace(1, 4, 7)).astype(int)
M_list = np.round(10**np.linspace(1, 4, 7)).astype(int)[:-1]
N_list = np.round(10**np.linspace(1, np.log10(1548), 7)).astype(int)

# Fix N
N_fix = N_list[-1]
errors_M_sweep = np.zeros((len(J_list),len(M_list),2,idx_max + 1))
for idx in range(idx_max + 1):
    for i, J in enumerate(J_list):
        for j, M in enumerate(M_list):
            errors_M_sweep[i,j,...,idx] = np.load(pref + 'idx' + str(idx)
                                              + '_J' + str(J)
                                              + 'm' + str(M)
                                              + 'n' + str(N_fix)
                                              + '/' + er_fname)**2

errors_M_sweep_std = np.std(errors_M_sweep,axis=-1)
errors_M_sweep_mean = np.mean(errors_M_sweep,axis=-1)
lb = errors_M_sweep_mean - n_std*errors_M_sweep_std
ub = errors_M_sweep_mean + n_std*errors_M_sweep_std

plt.figure(0)
plt.loglog(M_list, errors_M_sweep_mean[...,0].T, 'o:')
for i in range(len(J_list)):
    plt.fill_between(M_list, lb[i,...,0], ub[i,...,0], alpha=0.2)
plt.loglog(M_list, 1./M_list**1)
plt.xlabel(r'$M$')
plt.ylabel(r'Relative Bochner squared error')
plt.grid(axis='both')
if FLAG_SAVE:
    plt.savefig('figures/' + 'Nfix' + str(N_fix) + '_boch2_temp' + '.pdf', format='pdf')

# plt.figure(1)
# plt.loglog(M_list, errors_M_sweep[...,1].T, 'o:')
# plt.loglog(M_list, 1./M_list**.5)
# plt.xlabel(r'$M$')
# plt.ylabel(r'Relative Bochner error')
# plt.grid(axis='both')
# if FLAG_SAVE:
#     plt.savefig('figures/' + 'Nfix' + str(N_fix) + '_boch' + '.pdf', format='pdf')

# # Fix M
# M_fix = M_list[-1]
# errors_N_sweep = np.zeros((3,7,2))
# for i, J in enumerate(J_list):
#     for j, N in enumerate(N_list):
#         errors_N_sweep[i,j,...] = np.load(pref + 'idx' + str(idx)
#                                           + '_J' + str(J)
#                                           + 'm' + str(M_fix)
#                                           + 'n' + str(N)
#                                           + '/' + er_fname)

# plt.figure(10)
# plt.loglog(N_list, errors_N_sweep[...,0].T, 'o:')
# plt.loglog(N_list, 1./N_list**.25)
# plt.xlabel(r'$N$')
# plt.ylabel(r'Average relative error')
# plt.grid(axis='both')
# if FLAG_SAVE:
#     plt.savefig('figures/' + 'Mfix' + str(M_fix) + '_rel_bigreg' + '.pdf', format='pdf')

# plt.figure(11)
# plt.loglog(N_list, errors_N_sweep[...,1].T, 'o:')
# plt.loglog(N_list, 1./N_list**.25)
# plt.xlabel(r'$N$')
# plt.ylabel(r'Relative Bochner error')
# plt.grid(axis='both')
# if FLAG_SAVE:
#     plt.savefig('figures/' + 'Mfix' + str(M_fix) + '_boch_bigreg' + '.pdf', format='pdf')

# plt.show()