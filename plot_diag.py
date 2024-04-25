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
al_fname = "al_model.npy"
suf = 'diag' # boch2_sweep
FLAG_SAVE = True
idx_max = 9
n_std = 2

style_list = ['C0d:', 'C1s-.', 'C2o--']
color_list = ['C0', 'C1', 'C2']
leg_list = [r'$p=256$', r'$p=1024$', r'$p=4096$']
ll = True
ll_al = not True
fsd = not True

J_list = np.asarray([256, 1024, 4096])
# M_list = np.round(10**np.linspace(1, 4, 7)).astype(int)
# N_list = np.round(10**np.linspace(1, np.log10(1548), 7)).astype(int)

# M_fix = M_list[-1] # [-1], [-3], [-5]
# N_fix = N_list[-1] # [-1] = 1548, N_list[-4] = 124

# M_list = np.round(10**np.linspace(1, 4, 7)).astype(int)[:-2]

# %% Diag M
N_list = np.asarray((2, 5, 16, 49, 155, 490, 1548))
M_list = np.asarray((100, 178, 316, 562, 1000, 1778, 3162))

# pref = '/home/nnelsen/code/error-bounds-for-vvRF/results_sweep_M_diag/vvRF_'
pref = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/results_sweep_M_diag/vvRF_'

alnorm_M_sweep = np.zeros((len(J_list),len(M_list),idx_max + 1))
errors_M_sweep = np.zeros((len(J_list),len(M_list),2,idx_max + 1))
for idx in range(idx_max + 1):
    for i, J in enumerate(J_list):
        for j, (M, N) in enumerate(zip(M_list, N_list)):
            alnorm_M_sweep[i,j,idx] = np.sqrt(np.mean(np.load(pref + 'idx' + str(idx)
                                          + '_J' + str(J)
                                          + 'm' + str(M)
                                          + 'n' + str(N)
                                          + '/' + al_fname)**2)) # M-norm of \hat{\alpha}
            
            if er_fname == "errors_reg.npy":
                errors_M_sweep[i,j,...,idx] = np.min(np.load(pref + 'idx' + str(idx)
                                                  + '_J' + str(J)
                                                  + 'm' + str(M)
                                                  + 'n' + str(N)
                                                  + '/' + er_fname), 
                                                     axis=0)[-2:]**2
            else:
                errors_M_sweep[i,j,...,idx] = np.load(pref + 'idx' + str(idx)
                                              + '_J' + str(J)
                                              + 'm' + str(M)
                                              + 'n' + str(N)
                                              + '/' + er_fname)**2

alnorm_M_sweep_std = np.std(alnorm_M_sweep,axis=-1)
alnorm_M_sweep_mean = np.mean(alnorm_M_sweep,axis=-1)
lb_al = alnorm_M_sweep_mean - n_std*alnorm_M_sweep_std
ub_al = alnorm_M_sweep_mean + n_std*alnorm_M_sweep_std

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
    plotter.save_plot(f0, 'figures/' + 'Mdiag_M' + '_' + suf + '.pdf')

idx_plot_M = 10
for i in range(len(J_list)):
    plotter.plot_oneD(idx_plot_M, N_list, errors_M_sweep_mean[i,...,-1], linestyle_str=style_list[i], loglog=ll, legendlab_str=leg_list[i], fig_sz_default=fsd)
    plt.fill_between(N_list, lb[i,...,-1], ub[i,...,-1], facecolor=color_list[i], alpha=0.2)
f0 = plotter.plot_oneD(idx_plot_M, N_list, 1e-1*N_list**(-0.5), xlab_str1D=r'$N$', ylab_str1D=r'Relative Bochner squared error', linestyle=(0, (3, 1, 1, 1, 1, 1)), linestyle_str='darkgray', legendlab_str=r'$N^{-1/2}$', fig_sz_default=fsd)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
if FLAG_SAVE:
    plotter.save_plot(f0, 'figures/' + 'Mdiag_N' + '_' + suf + '.pdf')
    
# Plot alpha hat
idx_plot_M = 2
for i in range(len(J_list)):
    f0 = plotter.plot_oneD(idx_plot_M, M_list, alnorm_M_sweep_mean[i,...], linestyle_str=style_list[i], loglog=ll_al, legendlab_str=leg_list[i], fig_sz_default=fsd, xlab_str1D=r'$M$', ylab_str1D=r'$\|\widehat{\alpha}\|_M$')
    plt.fill_between(M_list, lb_al[i,...], ub_al[i,...], facecolor=color_list[i], alpha=0.2)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
plt.xscale('log')
if FLAG_SAVE:
    plotter.save_plot(f0, 'figures/' + 'Mdiag_M_alpha' + '_' + suf + '.pdf')

idx_plot_M = 20
for i in range(len(J_list)):
    f0 = plotter.plot_oneD(idx_plot_M, N_list, alnorm_M_sweep_mean[i,...], linestyle_str=style_list[i], loglog=ll_al, legendlab_str=leg_list[i], fig_sz_default=fsd, xlab_str1D=r'$N$', ylab_str1D=r'$\|\widehat{\alpha}\|_{M_N}$')
    plt.fill_between(N_list, lb_al[i,...], ub_al[i,...], facecolor=color_list[i], alpha=0.2)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
plt.xscale('log')
if FLAG_SAVE:
    plotter.save_plot(f0, 'figures/' + 'Mdiag_N_alpha' + '_' + suf + '.pdf')
    
# idx_plot_M = 10
# plt.figure(idx_plot_M, figsize=(5.1667,5.1667))
# for i in range(len(M_list)):
#     plt.loglog(J_list, errors_M_sweep_mean[...,i,-1], 'o:', label=r'$M=%d$' %(M_list[i]))
#     plt.fill_between(J_list, lb[...,i,-1], ub[...,i,-1], alpha=0.2)
# plt.xlabel(r'$p$')
# plt.ylabel(r'Relative Bochner squared error')
# plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
# plt.grid(axis='both')
# if FLAG_SAVE:
#     plt.savefig('figures/' + 'Mdiag_M_p_' + '_' + suf + '.pdf', format='pdf', bbox_inches='tight')


# %% Diag N
N_list = np.asarray((10, 23, 54, 124, 288, 668, 1548))
M_list = np.asarray((254, 385, 591, 895, 1364, 2077, 3162))

# pref = '/home/nnelsen/code/error-bounds-for-vvRF/results_sweep_N_diag/vvRF_'
pref = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/results_sweep_N_diag/vvRF_'

alnorm_N_sweep = np.zeros((len(J_list),len(N_list),idx_max + 1))
errors_N_sweep = np.zeros((len(J_list),len(N_list),2,idx_max + 1))
for idx in range(idx_max + 1):
    for i, J in enumerate(J_list):
        for j, (M, N) in enumerate(zip(M_list, N_list)):
            alnorm_N_sweep[i,j,idx] = np.sqrt(np.mean(np.load(pref + 'idx' + str(idx)
                                          + '_J' + str(J)
                                          + 'm' + str(M)
                                          + 'n' + str(N)
                                          + '/' + al_fname)**2)) # M-norm of \hat{\alpha}
            
            if er_fname == "errors_reg.npy":
                errors_N_sweep[i,j,...,idx] = np.min(np.load(pref + 'idx' + str(idx)
                                                  + '_J' + str(J)
                                                  + 'm' + str(M)
                                                  + 'n' + str(N)
                                                  + '/' + er_fname), 
                                                     axis=0)[-2:]**2
            else:
                errors_N_sweep[i,j,...,idx] = np.load(pref + 'idx' + str(idx)
                                              + '_J' + str(J)
                                              + 'm' + str(M)
                                              + 'n' + str(N)
                                              + '/' + er_fname)**2

alnorm_N_sweep_std = np.std(alnorm_N_sweep,axis=-1)
alnorm_N_sweep_mean = np.mean(alnorm_N_sweep,axis=-1)
lb_al = alnorm_N_sweep_mean - n_std*alnorm_N_sweep_std
ub_al = alnorm_N_sweep_mean + n_std*alnorm_N_sweep_std

errors_N_sweep_std = np.std(errors_N_sweep,axis=-1)
errors_N_sweep_mean = np.mean(errors_N_sweep,axis=-1)
lb = errors_N_sweep_mean - n_std*errors_N_sweep_std
ub = errors_N_sweep_mean + n_std*errors_N_sweep_std

idx_plot_N = 1
for i in range(len(J_list)):
    plotter.plot_oneD(idx_plot_N, N_list, errors_N_sweep_mean[i,...,-1], linestyle_str=style_list[i], loglog=ll, legendlab_str=leg_list[i], fig_sz_default=fsd)
    plt.fill_between(N_list, lb[i,...,-1], ub[i,...,-1], facecolor=color_list[i], alpha=0.2)
f1 = plotter.plot_oneD(idx_plot_N, N_list, 2.5e-2*N_list**(-0.5), xlab_str1D=r'$N$', ylab_str1D=r'Relative Bochner squared error', linestyle=(0, (3, 1, 1, 1, 1, 1)), linestyle_str='darkgray', legendlab_str=r'$N^{-1/2}$', fig_sz_default=fsd)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
if FLAG_SAVE:
    plotter.save_plot(f1, 'figures/' + 'Ndiag_N' + '_' + suf + '.pdf')

idx_plot_N = 11
for i in range(len(J_list)):
    plotter.plot_oneD(idx_plot_N, M_list, errors_N_sweep_mean[i,...,-1], linestyle_str=style_list[i], loglog=ll, legendlab_str=leg_list[i], fig_sz_default=fsd)
    plt.fill_between(M_list, lb[i,...,-1], ub[i,...,-1], facecolor=color_list[i], alpha=0.2)
f1 = plotter.plot_oneD(idx_plot_N, M_list, 1.5e0*M_list**(-1.0), xlab_str1D=r'$M$', ylab_str1D=r'Relative Bochner squared error', linestyle=(0, (3, 1, 1, 1, 1, 1)), linestyle_str='darkgray', legendlab_str=r'$M^{-1}$', fig_sz_default=fsd)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
if FLAG_SAVE:
    plotter.save_plot(f1, 'figures/' + 'Ndiag_M' + '_' + suf + '.pdf')
    
# Plot alpha hat
idx_plot_N = 3
for i in range(len(J_list)):
    f1 = plotter.plot_oneD(idx_plot_N, N_list, alnorm_N_sweep_mean[i,...], linestyle_str=style_list[i], loglog=ll_al, legendlab_str=leg_list[i], fig_sz_default=fsd, xlab_str1D=r'$N$', ylab_str1D=r'$\|\widehat{\alpha}\|_{M_N}$')
    plt.fill_between(N_list, lb_al[i,...], ub_al[i,...], facecolor=color_list[i], alpha=0.2)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
plt.xscale('log')
if FLAG_SAVE:
    plotter.save_plot(f1, 'figures/' + 'Ndiag_N_alpha' + '_' + suf + '.pdf')

idx_plot_N = 30
for i in range(len(J_list)):
    f1 = plotter.plot_oneD(idx_plot_N, M_list, alnorm_N_sweep_mean[i,...], linestyle_str=style_list[i], loglog=ll_al, legendlab_str=leg_list[i], fig_sz_default=fsd, xlab_str1D=r'$M$', ylab_str1D=r'$\|\widehat{\alpha}\|_{M}$')
    plt.fill_between(M_list, lb_al[i,...], ub_al[i,...], facecolor=color_list[i], alpha=0.2)
plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.grid(axis='both')
plt.xscale('log')
if FLAG_SAVE:
    plotter.save_plot(f1, 'figures/' + 'Ndiag_M_alpha' + '_' + suf + '.pdf')

# plt.figure(idx_plot_N, figsize=(5.1667,5.1667))
# for i in range(len(N_list)):
#     plt.loglog(J_list, errors_N_sweep_mean[...,i,-1], 'o:', label=r'$N=%d$' %(N_list[i]))
#     plt.fill_between(J_list, lb[...,i,-1], ub[...,i,-1], alpha=0.2)
# plt.xlabel(r'$p$')
# plt.ylabel(r'Relative Bochner squared error')
# plt.legend(loc='best',borderpad=borderpad,handlelength=handlelength).set_draggable(True)
# plt.grid(axis='both')
# if FLAG_SAVE:
#     plt.savefig('figures/' + 'Mfix_p_' + str(M_fix) + '_' + suf + '.pdf', format='pdf', bbox_inches='tight')