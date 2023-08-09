import numpy as np
import matplotlib.pyplot as plt
from utilities.plot_suiteSIAM import Plotter
plotter = Plotter() # set plotter class

plt.rcParams['font.size'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rc('legend', fontsize=14)

handlelength = 2.75
borderpad = 0.15

plt.close("all")

# er_fname = "errors_reg.npy"
er_fname = "errors_test_r_b.npy"
suf = 'boch2_rebuttal' # boch2_sweep
FLAG_SAVE = True
idx_max = 9
n_std = 2

style_list = ['C0d:', 'C1s-.', 'C2o--']
color_list = ['C0', 'C1', 'C2']
leg_list = [r'$p=256$', r'$p=1024$', r'$p=4096$']
ll = True
fsd = not True

J_list = np.asarray([256, 1024, 4096])
M_list = np.round(10**np.linspace(1, 4, 7)).astype(int)
N_list = np.round(10**np.linspace(1, np.log10(1548), 7)).astype(int)

M_fix = M_list[-1] # [-1], [-3], [-5]
N_fix = N_list[-1] # [-1] = 1548, N_list[-4] = 124

M_list = np.round(10**np.linspace(1, 4, 7)).astype(int)[:-2]

# %% Fix N
pref = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/results/vvRF_'
# pref = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/results_sweep_M/vvRF_'
errors_M_sweep = np.zeros((len(J_list),len(M_list),2,idx_max + 1))
for idx in range(idx_max + 1):
    for i, J in enumerate(J_list):
        for j, M in enumerate(M_list):
            if er_fname == "errors_reg.npy":
                errors_M_sweep[i,j,...,idx] = np.min(np.load(pref + 'idx' + str(idx)
                                                  + '_J' + str(J)
                                                  + 'm' + str(M)
                                                  + 'n' + str(N_fix)
                                                  + '/' + er_fname), 
                                                     axis=0)[-2:]**2
            else:
                errors_M_sweep[i,j,...,idx] = np.load(pref + 'idx' + str(idx)
                                              + '_J' + str(J)
                                              + 'm' + str(M)
                                              + 'n' + str(N_fix)
                                              + '/' + er_fname)**2

errors_M_sweep_std = np.std(errors_M_sweep,axis=-1)
errors_M_sweep_mean = np.mean(errors_M_sweep,axis=-1)
lb = errors_M_sweep_mean - n_std*errors_M_sweep_std
ub = errors_M_sweep_mean + n_std*errors_M_sweep_std

idx_plot_M = 0
for i in range(len(J_list)):
    plotter.plot_oneD(idx_plot_M, M_list, errors_M_sweep_mean[i,...,-1], linestyle_str=style_list[i], loglog=ll, legendlab_str=leg_list[i], fig_sz_default=fsd)
    plt.fill_between(M_list, lb[i,...,-1], ub[i,...,-1], facecolor=color_list[i], alpha=0.2)
f0 = plotter.plot_oneD(idx_plot_M, M_list, M_list**(-1.0), xlab_str1D=r'$M$', ylab_str1D=r'Relative Bochner squared error', linestyle=(0, (3, 1, 1, 1, 1, 1)), linestyle_str='darkgray', legendlab_str=r'$M^{-1}$', fig_sz_default=fsd)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
if FLAG_SAVE:
    plotter.save_plot(f0, 'figures/' + 'Nfix' + str(N_fix) + '_' + suf + '.pdf')
    
idx_plot_M = 10
plt.figure(idx_plot_M, figsize=(5.1667,5.1667))
for i in range(len(M_list)):
    plt.loglog(J_list, errors_M_sweep_mean[...,i,-1], 'o:', label=r'$M=%d$' %(M_list[i]))
    plt.fill_between(J_list, lb[...,i,-1], ub[...,i,-1], alpha=0.2)
plt.xlabel(r'$p$')
plt.ylabel(r'Relative Bochner squared error')
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
if FLAG_SAVE:
    plt.savefig('figures/' + 'Nfix_p_' + str(N_fix) + '_' + suf + '.pdf', format='pdf', bbox_inches='tight')


# Fix M
# pref = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/results/vvRF_'
pref = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/results_sweep_N/vvRF_'
errors_N_sweep = np.zeros((len(J_list),len(N_list),2,idx_max + 1))
for idx in range(idx_max + 1):
    for i, J in enumerate(J_list):
        for j, N in enumerate(N_list):
            if er_fname == "errors_reg.npy":
                errors_N_sweep[i,j,...,idx] = np.min(np.load(pref + 'idx' + str(idx)
                                                  + '_J' + str(J)
                                                  + 'm' + str(M_fix)
                                                  + 'n' + str(N)
                                                  + '/' + er_fname), 
                                                     axis=0)[-2:]**2
            else:
                errors_N_sweep[i,j,...,idx] = np.load(pref + 'idx' + str(idx)
                                              + '_J' + str(J)
                                              + 'm' + str(M_fix)
                                              + 'n' + str(N)
                                              + '/' + er_fname)**2

errors_N_sweep_std = np.std(errors_N_sweep,axis=-1)
errors_N_sweep_mean = np.mean(errors_N_sweep,axis=-1)
lb = errors_N_sweep_mean - n_std*errors_N_sweep_std
ub = errors_N_sweep_mean + n_std*errors_N_sweep_std

idx_plot_N = 1
for i in range(len(J_list)):
    plotter.plot_oneD(idx_plot_N, N_list, errors_N_sweep_mean[i,...,-1], linestyle_str=style_list[i], loglog=ll, legendlab_str=leg_list[i], fig_sz_default=fsd)
    plt.fill_between(N_list, lb[i,...,-1], ub[i,...,-1], facecolor=color_list[i], alpha=0.2)
f1 = plotter.plot_oneD(idx_plot_N, N_list, 1.5e-2*N_list**(-0.5), xlab_str1D=r'$N$', ylab_str1D=r'Relative Bochner squared error', linestyle=(0, (3, 1, 1, 1, 1, 1)), linestyle_str='darkgray', legendlab_str=r'$N^{-1/2}$', fig_sz_default=fsd)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
if FLAG_SAVE:
    plotter.save_plot(f1, 'figures/' + 'Mfix' + str(M_fix) + '_' + suf + '.pdf')

idx_plot_N = 11
plt.figure(idx_plot_N, figsize=(5.1667,5.1667))
for i in range(len(N_list)):
    plt.loglog(J_list, errors_N_sweep_mean[...,i,-1], 'o:', label=r'$N=%d$' %(N_list[i]))
    plt.fill_between(J_list, lb[...,i,-1], ub[...,i,-1], alpha=0.2)
plt.xlabel(r'$p$')
plt.ylabel(r'Relative Bochner squared error')
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
if FLAG_SAVE:
    plt.savefig('figures/' + 'Mfix_p_' + str(M_fix) + '_' + suf + '.pdf', format='pdf', bbox_inches='tight')