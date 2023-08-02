import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime

import os, sys
from utilities_module import DataReader
from RFM_UQ import RandomFeatureModel, InnerProduct1D, UQ
from utilities_module import truncate_rfft2, half_to_full
from utilities.plot_suiteSIAM import Plotter

# Output directory
def make_save_path(test_str, pth = "/VVRF_"):
    save_path = "results/" + datetime.today().strftime('%Y-%m-%d') + pth + test_str +"/"
    return save_path

# %% Problem Setup

# Process command line arguments
print(sys.argv)
m = int(sys.argv[1])                    # number of features
n = int(sys.argv[2])                    # training sample size
var_pow = int(sys.argv[3])              # 10^-(var_pow) is the noise variance

# Logistic
data_path = "data/burgers_data_R10.mat"                     # path to dataset
TEST_STR = "var1Em" + str(var_pow) + "_all"
user_comment = "UQ_paper_runs_HPC"
FLAG_SAVE = True
FLAG_PW = True
FLAG_JOINT = True
newseed = None              # e.g.: None or int(datetime.today().strftime('%Y%m%d'))

# Problem
var_noise = 10.0**(-var_pow)
lamreg = var_noise
K = 256                     # resolution, must be power of two
ntest = 100                 # testing sample size (maximum is 2048 - n)
bsize_train = 50
bsize_test = 50
bsize_grf_train = 100
bsize_grf_test = 100
bsize_grf_sample = 500
kmax = 64                   # zero pad after kmax RF GRF modes, kmax <= K//2

# %% Pre-trained hyperparameters for Burgers' dataset (using alternating LBFGS)

# ##### Option 1
# sig_rf = 5.5253
# nu_rf = 4e-4
# al_rf = 0.76
# sig_g = 51.1982
# tau_g = 8.07
# al_g = 2.7223

# ##### Option 2
sig_rf = 2.597733
nu_rf = 0.31946787
al_rf = 0.1
sig_g = 1.7861565
tau_g = 15.045227
al_g = 2.9943917

# %% File I/O
if TEST_STR is None:
    TEST_STR = "m" + str(m) +"n" + str(n)
else:
    TEST_STR = TEST_STR + "_" + "m" + str(m) +"n" + str(n)
    
save_path = make_save_path(TEST_STR)

os.makedirs(save_path, exist_ok=True)

# %% Process data
if n + ntest > 2048:
    raise ValueError("ERROR: n + ntest must be less than 2048 + 1 for the Burgers' dataset.")

# Load training data and test data
reader = DataReader(data_path)
input_train = reader.read_field("a")
K_fine = input_train.shape[-1]
width_subsample = round(K_fine/K)
input_train = input_train[..., ::width_subsample]
output_train = reader.read_field("u")[..., ::width_subsample]
    
# shuffle
dataset_shuffle_idx = torch.randperm(input_train.shape[0])
input_train = input_train[dataset_shuffle_idx, ...]
output_train = output_train[dataset_shuffle_idx, ...]

# extract
input_test = input_train[-ntest:, ...]
input_train = input_train[:n, ...]
output_test = output_train[-ntest:, ...]
output_train = output_train[:n, ...]
input_train = input_train
output_train = output_train
input_test = input_test
output_test = output_test

# add noise to outputs
if var_noise != 0:
    output_train_noisy = output_train +\
                            np.sqrt(np.abs(var_noise))*torch.randn(output_train.shape)
else:
    output_train_noisy = output_train

# %% Setup model and save hyperparameters
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if newseed is not None:
    np.random.seed(newseed)
    torch.manual_seed(newseed)
            
rfm = RandomFeatureModel(K, n, m, ntest, lamreg, nu_rf, al_rf, bsize_train=bsize_train, bsize_test=bsize_test, bsize_grf_train=bsize_grf_train, bsize_grf_test=bsize_grf_test, bsize_grf_sample=bsize_grf_sample, device=dev, al_g=al_g, tau_g=tau_g, sig_g=sig_g, kmax=kmax, sig_rf=sig_rf, var_noise=var_noise, K_fine=K_fine)

rfm.load_train(input_train, output_train)
rfm.load_test(input_test, output_test)
rfm.output_train_noisy = output_train_noisy

hyp_array = np.asarray((user_comment,newseed,K,m,n,ntest,bsize_train,bsize_test,bsize_grf_train,bsize_grf_test,bsize_grf_sample,kmax,sig_rf,nu_rf,al_rf,sig_g,tau_g,al_g,lamreg,dataset_shuffle_idx,var_noise), dtype=object) # array of hyperparameters

# Write hyperparameters to file
if FLAG_SAVE:
    np.save(save_path + "hyperparameters.npy", hyp_array)

# %% Least squares train and regularization grid sweep
start = time.time() 
rfm.fit()
if var_noise == 0:
    e_reg = rfm.regsweep([1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5])
print('Total Time Elapsed: ', time.time() - start, 'seconds.') # print run time
print('\n RKHS Norm of coeff:', torch.linalg.norm(rfm.al_model.cpu()).item()/np.sqrt(rfm.m),'; Max coeff:', torch.max(torch.abs(rfm.al_model.cpu())).item())

# %% Global expected relative train and test errors
print("\n Global expected relative train and test errors:")
start = time.time() 
e_train, b_train, errors_train = rfm.relative_error_train(FLAG_GETERRORS=True)
print('Expected relative error (Train, Boch. Train):', (e_train, b_train))
e_test, b_test, errors_test = rfm.relative_error_test(FLAG_GETERRORS=True)
print('Expected relative error (Test, Boch. Test):', (e_test, b_test))
print('Total Train and Test Error Time Elapsed: ', time.time() - start, 'seconds.') # print run time

# %% Save to file
if FLAG_SAVE:
    np.save(save_path + 'errors_train.npy', errors_train.cpu().numpy())
    np.save(save_path + 'errors_test.npy', errors_test.cpu().numpy())
    
    np.save(save_path + 'grf_g.npy', rfm.grf_g.cpu().numpy())
    np.save(save_path + 'al_model.npy', rfm.al_model.cpu().numpy())

# %% Setup for plotting results

# USER INPUT
nstd = 1.96
n_samples = 5
k_trunc_pw = 40
k_trunc_joint = 32
lamreg = rfm.lamreg
FLAG_TRAIN_IDX = False

# Derived
input_train, output_train = rfm.input_train, rfm.output_train
input_test, output_test = rfm.input_test, rfm.output_test
uq = UQ(rfm, lamreg=lamreg)
ip = InnerProduct1D(uq.rfm.h)

# Store variances
pw_var_train = torch.zeros(input_train.shape) # normalized pw variance function
pw_var_test = torch.zeros(input_test.shape)
in_batches = torch.split(input_train, rfm.bsize_test)
tin_batches = torch.split(input_test, rfm.bsize_test)
in_idxs = torch.split(torch.arange(input_train.shape[0]), rfm.bsize_test)
tin_idxs = torch.split(torch.arange(input_test.shape[0]), rfm.bsize_test)
for idx, in_batch in zip(in_idxs, in_batches):
    y_temp = rfm.predict(in_batch).cpu()
    pw_var_train[idx, ...] = uq.pw_cov_posterior(in_batch).cpu() / ip.L2(y_temp, y_temp)[:,None]
for idx, test_batch in zip(tin_idxs, tin_batches):
    y_temp = rfm.predict(test_batch).cpu()
    pw_var_test[idx, ...] = uq.pw_cov_posterior(test_batch).cpu() / ip.L2(y_temp, y_temp)[:,None]
pw_std_train_trace = np.sqrt(torch.trapz(pw_var_train, dx=rfm.h, dim=-1))
pw_std_test_trace = np.sqrt(torch.trapz(pw_var_test, dx=rfm.h, dim=-1))

# %% Main pointwise plotting function
def make_pw_plots(uq,
               a_test,
               y_test,
               n_samples=5,
               nstd=1.96,
               k_trunc=40,
               FLAG_saveplot=False,
               str_exptype="test"
               ):
    # Derived
    plotter = Plotter(xlab_str=r'$x$', ylab_str=r'') # set plotter class
    x = torch.arange(0, 1., 1./a_test.shape[-1])*2*np.pi
    if str_exptype is None:
        str_exptype = "test"
    
    # Compute posterior
    y_pred = uq.rfm.predict(a_test).cpu()
    samp_post, cov_pred = uq.sample_pw_posterior(a_test, n_samples, True)
    samp_post = samp_post.cpu().squeeze()
    cov_pred = cov_pred.cpu()
    var_pred = torch.diag(cov_pred)
    std_pred = torch.sqrt(torch.maximum(torch.tensor(0), var_pred))
    
    # Error
    res = y_test - y_pred
    print('\n Evaluation point relative error:' , np.sqrt(ip.L2(res,res)/ip.L2(y_test,y_test)).item())
    
    # Input/output plus 95% CI
    plotter.plot_oneD(1, x, a_test, legendlab_str=r'IC', linestyle_str='C5')
    ax = plt.gca()
    ax.fill_between(x, y_pred + nstd*std_pred, y_pred - nstd*std_pred, facecolor='k', alpha=0.2)
    plotter.plot_oneD(1, x, y_test, legendlab_str=r'Truth', linestyle_str='C3')
    if K > 65:
        f1 = plotter.plot_oneD(1, x, y_pred, legendlab_str=r'RFM', linestyle_str='k', linestyle=(0, (3, 1, 1, 1)))
        plt.legend(loc='best')
        f11 = plotter.plot_oneD(11, x, np.abs(res)**2, ylab_str1D=r'Squared Pointwise Error', linestyle_str='k')
    else:
        f1 = plotter.plot_oneD(1, x, y_pred, legendlab_str=r'RFM')
        plt.legend(loc='best')
        f11 = plotter.plot_oneD(11, x, np.abs(res)**2, ylab_str1D=r'Squared Pointwise Error')
        
    # Posterior samples, mean, truth, and 95% CI
    plotter.plot_oneD(2, x, y_test, legendlab_str=r'Truth', linestyle_str='k', LW_set=2.5, linestyle=(0, (3, 1, 1, 1)))
    ax = plt.gca()
    ax.fill_between(x, y_pred + nstd*std_pred, y_pred - nstd*std_pred, facecolor='k', alpha=0.2)
    for samp in range(n_samples):
        plt.plot(x, samp_post[samp, :])
    if K > 65:
        f2 = plotter.plot_oneD(2, x, y_pred, legendlab_str=r'Mean', linestyle_str='k-', LW_set=3)
    else:
        f2 = plotter.plot_oneD(2, x, y_pred, legendlab_str=r'Mean', LW_set=3)
    plt.legend(loc='best')
    
    # Pointwise posterior variance
    f22 = plotter.plot_oneD(22, x, var_pred, linestyle_str='C0', ylab_str1D="Pointwise Variance")
    
    # Posterior covariance operator (its kernel function) and its fft
    x1, x2 = torch.meshgrid(x, x)
    plotter2d = Plotter(xlab_str=r'', ylab_str=r'') # set plotter class
    f23 = plotter2d.plot_Heat(23, x1, x2, cov_pred)
    fcov_imag = torch.fft.rfft2(cov_pred, norm='forward')
    fcov_imag = half_to_full(truncate_rfft2(fcov_imag, k_trunc//2))
    fcov_imag = torch.fft.fftshift(fcov_imag)
    fcov_real = fcov_imag.real
    fcov_imag = fcov_imag.imag
    ext = (-k_trunc//2, k_trunc//2, -k_trunc//2, k_trunc//2)
    f24 = plt.figure(24, figsize=(9, 9))
    plt.subplot(1,2,1)
    plt.imshow(fcov_real, origin='lower', interpolation='none', extent=ext)
    plt.box(False)
    plt.subplot(1,2,2)
    plt.imshow(fcov_imag, origin='lower', interpolation='none', extent=ext)
    plt.box(False)
    plt.tight_layout()
    
    # Compute prior
    samp_prior, cov_prior = uq.sample_pw_prior(a_test, n_samples, True)
    samp_prior = samp_prior.cpu().squeeze()
    cov_prior = cov_prior.cpu()
    var_prior = torch.diag(cov_prior)
    std_prior = torch.sqrt(torch.maximum(torch.tensor(0), var_prior))
    
    # Prior samples, mean, and 95% CI
    if K > 65:
        f3 = plotter.plot_oneD(3, x, y_pred*0.0, legendlab_str=r'Mean', linestyle_str='k-', LW_set=3)
    else:
        f3 = plotter.plot_oneD(3, x, y_pred*0.0, legendlab_str=r'Mean', LW_set=3)
    for samp in range(n_samples):
        plt.plot(x, samp_prior[samp, :])
    ax = plt.gca()
    ax.fill_between(x, nstd*std_prior, -nstd*std_prior, facecolor='k', alpha=0.2)
    plt.legend(loc='best')
    
    # Pointwise prior variance
    f32 = plotter.plot_oneD(32, x, var_prior, linestyle_str='C0', ylab_str1D="Pointwise Variance")
    
    # Prior covariance operator (its kernel function) and its fft
    f33 = plotter2d.plot_Heat(33, x1, x2, cov_prior)
    pfcov_imag = torch.fft.rfft2(cov_prior, norm='forward')
    pfcov_imag = half_to_full(truncate_rfft2(pfcov_imag, k_trunc//2))
    pfcov_imag = torch.fft.fftshift(pfcov_imag)
    pfcov_real = pfcov_imag.real
    pfcov_imag = pfcov_imag.imag
    f34 = plt.figure(34, figsize=(9, 9))
    plt.subplot(1,2,1)
    plt.imshow(pfcov_real, origin='lower', interpolation='none', extent=ext)
    plt.box(False)
    plt.subplot(1,2,2)
    plt.imshow(pfcov_imag, origin='lower', interpolation='none', extent=ext)
    plt.box(False)
    plt.tight_layout()

    # Save current figs to file
    if FLAG_saveplot:
        plotter.save_plot(f1, save_path + "io_ci_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        plotter.save_plot(f11, save_path + "pwe_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        
        plotter.save_plot(f2, save_path + "samp_post_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        plotter.save_plot(f23, save_path + "cov_post_" + str_exptype + ".png", fig_format="png", bbox="tight")
        plotter.save_plot(f24, save_path + "cov_fft_post_" + str_exptype + ".png", fig_format="png", bbox="tight")
        plotter.save_plot(f22, save_path + "var_post_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        
        plotter.save_plot(f3, save_path + "samp_prior_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        plotter.save_plot(f33, save_path + "cov_prior_" + str_exptype + ".png", fig_format="png", bbox="tight")
        plotter.save_plot(f34, save_path + "cov_fft_prior_" + str_exptype + ".png", fig_format="png", bbox="tight")
        plotter.save_plot(f32, save_path + "var_prior_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        
# %% Main joint plotting function
def make_joint_plots(uq,
               a_batch,
               y_batch,
               n_samples=5,
               nstd=1.96,
               k_trunc=32,
               FLAG_saveplot=False,
               str_exptype="test",
               max_loop_plot=5,
               FLAG_OWMB=True
               ):
    # Derived
    plotter = Plotter(xlab_str=r'$x$', ylab_str=r'') # set plotter class
    x = torch.arange(0, 1., 1./a_batch.shape[-1])*2*np.pi
    if str_exptype is None:
        str_exptype = "test"
    
    # Compute posterior
    y_pred = uq.rfm.predict(a_batch).cpu()
    samp_post, cov_joint = uq.sample_joint_posterior(a_batch, n_samples, True)
    samp_post = samp_post.squeeze()
    var_joint = torch.diagonal(cov_joint, dim1=-2, dim2=-1)
    std_joint = torch.sqrt(torch.maximum(torch.tensor(0.), var_joint))
    std_pred = torch.diagonal(std_joint, dim1=0, dim2=1).permute(1,0)
    
    # Posterior samples, mean, truth, and 95% CI
    plt.close('all')
    for f_idx, y in enumerate(y_batch):
        if f_idx + 1 > max_loop_plot:
            break
        else:
            yp = y_pred[f_idx, ...]
            st = std_pred[f_idx, ...]
            plotter.plot_oneD(f_idx, x, y, legendlab_str=r'Truth', linestyle_str='k', LW_set=2.5, linestyle=(0, (3, 1, 1, 1)))
            ax = plt.gca()
            ax.fill_between(x, yp + nstd*st, yp - nstd*st, facecolor='k', alpha=0.2)
            for samp in samp_post[:,f_idx,...]:
                plt.plot(x, samp)
            if K > 65:
                f = plotter.plot_oneD(f_idx, x, yp, legendlab_str=r'Mean', linestyle_str='k-', LW_set=3)
            else:
                f = plotter.plot_oneD(f_idx, x, yp, legendlab_str=r'Mean', LW_set=3)
            plt.legend(loc='best')
            if FLAG_saveplot:
                plotter.save_plot(f, save_path + "samp_joint_post_" + str(f_idx) + "_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
            plt.close(f_idx)
    
    # Cross posterior pw variances
    if FLAG_OWMB:
        f = plt.figure(1)
        lbs = ["OOD", "Max", "Median", "Min"]
        for ii, v in enumerate(torch.diagonal(var_joint).permute(1,0)):
            yy = y_pred[ii, ...]
            plt.semilogy(x, v/ip.L2(yy, yy), label=lbs[ii])
        plt.xlabel(r"$x$")
        plt.ylabel("Normalized Variance")
        plt.legend(loc='best')
        if FLAG_saveplot:
            plotter.save_plot(f, save_path + "var_pwpw_post_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        plt.close(1)
    
    f = plt.figure(1)
    vidx = torch.triu_indices(*[var_joint.shape[0]]*2, 1)
    var_joint = var_joint[vidx[0], vidx[1], ...]
    for v in var_joint:
        plt.plot(x, v)
    plt.xlabel(r"$x$")
    plt.ylabel("Covariance")
    if FLAG_saveplot:
        plotter.save_plot(f, save_path + "var_crosspw_post_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
    plt.close(1)
    
    # Posterior covariance operator (its kernel function)
    x1, x2 = torch.meshgrid(x, x)
    plotter2d = Plotter(xlab_str=r'', ylab_str=r'') # set plotter class
    
    f = plotter2d.plot_Heat(1, x1, x2, uq.get_block_matrix(cov_joint, True), cb_ticks_sn=True)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_joint_post_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)
    
    f = plotter2d.plot_Heat(1, x1, x2, uq.get_block_matrix(cov_joint, False), cb_ticks_sn=True, interp_set='none')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_joint_mix_post_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)
    
    # FFT post
    fcov_imag = torch.fft.rfft2(cov_joint, norm='forward')
    fcov_imag = half_to_full(truncate_rfft2(fcov_imag, k_trunc//2))
    fcov_imag = torch.fft.fftshift(fcov_imag)
    fcov_real = fcov_imag.real
    fcov_imag = fcov_imag.imag
    ext = (-k_trunc//2, k_trunc//2, -k_trunc//2, k_trunc//2)
    
    f = plt.figure(1, figsize=(9, 9))
    plt.subplot(1,2,1)
    plt.imshow(uq.get_block_matrix(fcov_real, True), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplot(1,2,2)
    plt.imshow(uq.get_block_matrix(fcov_imag, True), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_fft_joint_post_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)
    
    f = plt.figure(1, figsize=(9, 9))
    plt.subplot(1,2,1)
    plt.imshow(uq.get_block_matrix(fcov_real, False), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplot(1,2,2)
    plt.imshow(uq.get_block_matrix(fcov_imag, False), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_fft_joint_mix_post_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)

    
    # Compute prior
    samp_prior, cov_prior = uq.sample_joint_prior(a_batch, n_samples, True)
    samp_prior = samp_prior.squeeze()
    var_prior = torch.diagonal(cov_prior, dim1=-2, dim2=-1)
    std_prior = torch.sqrt(torch.maximum(torch.tensor(0.), var_prior))
    std_prior = torch.diagonal(std_prior, dim1=0, dim2=1).permute(1,0)
    
    # Prior samples, mean, and 95% CI
    for f_idx, y in enumerate(y_batch):
        if f_idx + 1 > max_loop_plot:
            break
        else:
            st = std_prior[f_idx, ...]
            if K > 65:
                f = plotter.plot_oneD(f_idx, x, y*0., legendlab_str=r'Mean', linestyle_str='k-', LW_set=3)
            else:
                f = plotter.plot_oneD(f_idx, x, y*0., legendlab_str=r'Mean', LW_set=3)
            for samp in samp_prior[:,f_idx,...]:
                plt.plot(x, samp)
            ax = plt.gca()
            ax.fill_between(x, nstd*st, - nstd*st, facecolor='k', alpha=0.2)
            plt.legend(loc='best')
            if FLAG_saveplot:
                plotter.save_plot(f, save_path + "samp_joint_prior_" + str(f_idx) + "_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
            plt.close(f_idx)
    
    # Cross prior pw variances
    if FLAG_OWMB:
        f = plt.figure(1)
        lbs = ["OOD", "Max", "Median", "Min"]
        for ii, v in enumerate(torch.diagonal(var_prior).permute(1,0)):
            plt.plot(x, v, label=lbs[ii])
        plt.xlabel(r"$x$")
        plt.ylabel("Variance")
        plt.legend(loc='best')
        if FLAG_saveplot:
            plotter.save_plot(f, save_path + "var_pwpw_prior_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
        plt.close(1)
        
    f = plt.figure(1)
    vidx = torch.triu_indices(*[var_prior.shape[0]]*2, 1)
    var_prior = var_prior[vidx[0], vidx[1], ...]
    for v in var_prior:
        plt.plot(x, v)
    plt.xlabel(r"$x$")
    plt.ylabel("Covariance")
    if FLAG_saveplot:
        plotter.save_plot(f, save_path + "var_crosspw_prior_" + str_exptype + ".pdf", fig_format="pdf", bbox="tight")
    plt.close(1)
    
    # Prior covariance operator (its kernel function)
    x1, x2 = torch.meshgrid(x, x)
    plotter2d = Plotter(xlab_str=r'', ylab_str=r'') # set plotter class
    
    f = plotter2d.plot_Heat(1, x1, x2, uq.get_block_matrix(cov_prior, True), cb_ticks_sn=True)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_joint_prior_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)
    
    f = plotter2d.plot_Heat(1, x1, x2, uq.get_block_matrix(cov_prior, False), cb_ticks_sn=True, interp_set='none')
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_joint_mix_prior_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)

    # FFT prior
    fcov_imag = torch.fft.rfft2(cov_prior, norm='forward')
    fcov_imag = half_to_full(truncate_rfft2(fcov_imag, k_trunc//2))
    fcov_imag = torch.fft.fftshift(fcov_imag)
    fcov_real = fcov_imag.real
    fcov_imag = fcov_imag.imag
    ext = (-k_trunc//2, k_trunc//2, -k_trunc//2, k_trunc//2)
    
    f = plt.figure(1, figsize=(9, 9))
    plt.subplot(1,2,1)
    plt.imshow(uq.get_block_matrix(fcov_real, True), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplot(1,2,2)
    plt.imshow(uq.get_block_matrix(fcov_imag, True), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_fft_joint_prior_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)
    
    f = plt.figure(1, figsize=(9, 9))
    plt.subplot(1,2,1)
    plt.imshow(uq.get_block_matrix(fcov_real, False), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.subplot(1,2,2)
    plt.imshow(uq.get_block_matrix(fcov_imag, False), origin='lower', interpolation='none', extent=ext)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()
    if FLAG_saveplot:
        plotter2d.save_plot(f, save_path + "cov_fft_joint_mix_prior_" + str_exptype + ".png", fig_format="png", bbox="tight")
    plt.close(1)

# %% Make histogram plotting tool
def loghist(x, bins='auto', color='gray', alpha=0.65, edgecolor='k', density=True):
    """
    https://stackoverflow.com/questions/47850202/plotting-a-histogram-on-a-log-scale-with-matplotlib
    """
    _, bins = np.histogram(x, bins=bins)
    logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
    hr = plt.hist(x, bins=logbins, color=color, alpha=alpha, edgecolor=edgecolor, density=density)
    plt.xscale('log')
    return hr

def make_hist_plots(data,
                    data_owmb=None, # list when FLAG_OWMB is true
                    data_wmb_ref=None, # list when FLAG_OWMB is true
                    bins='auto',
                    color='gray',
                    alpha=0.65,
                    edgecolor='k',
                    density=True,
                    FLAG_saveplot=True,
                    str_exptype="test",
                    FLAG_OWMB=True,
                    FLAG_log=True,
                    xlabel=None
                    ):
    f0 = plt.figure(0)
    if FLAG_log:
        _ = loghist(data, bins=bins, color=color,
                        alpha=alpha, edgecolor=edgecolor, density=density)
    else:
        _ = plt.hist(data, bins=bins, color=color,
                        alpha=alpha, edgecolor=edgecolor, density=density)
    if FLAG_OWMB:
        lbs = ["OOD", "Max", "Median", "Min"]
        cols = ['C0', 'C1', 'C2', 'C3']
        for lab, col, val in zip(lbs, cols, data_owmb): 
            plt.axvline(val, linestyle='dashed', linewidth=3, label=lab, color=col)
        if data_wmb_ref is not None:
            for col, val in zip(cols[1:], data_wmb_ref): 
                plt.plot(val, 0., 'k^', markersize=18, markerfacecolor=col, markeredgewidth=1)
    plt.ylabel('Density')
    if xlabel is None:
        plt.xlabel(r"Normalized Posterior Standard Deviation")
    else:
        plt.xlabel(xlabel)
    plt.legend(loc='best').set_draggable(True)
    if FLAG_saveplot:
        f0.savefig(save_path + 'hist_' + str_exptype + '.pdf', format='pdf', bbox_inches='tight')
    plt.close(0)

# %% One random point
plt.close('all')

# Setup data
if FLAG_TRAIN_IDX:
    ind_test = np.random.randint(0, rfm.n)
    a_test = input_train[ind_test,:]
    y_test = output_train[ind_test,:]
else:
    ind_test = np.random.randint(0, rfm.ntest)
    a_test = input_test[ind_test,:]
    y_test = output_test[ind_test,:]

str_exptype = str(ind_test)

# make plots
if FLAG_PW:
    make_pw_plots(uq,
                a_test,
                y_test,
                n_samples=n_samples,
                nstd=nstd,
                k_trunc=k_trunc_pw,
                FLAG_saveplot=FLAG_SAVE,
                str_exptype=str_exptype
                )

# %% Worst, median, best (train)
plt.close('all')
idx_worst = torch.argmax(errors_train, dim=0).item()
idx_median = torch.argsort(errors_train)[errors_train.shape[0]//2, ...].item()
idx_best = torch.argmin(errors_train, dim=0).item()

idxs = [idx_worst, idx_median, idx_best]
names = ["worst_train", "median_train", "best_train"]

# loop over list
if FLAG_PW:
    for loop in range(len(idxs)):
        # Setup data
        ind_test = idxs[loop]
        a_test = input_train[ind_test,:]
        y_test = output_train[ind_test,:]
        
        str_exptype = names[loop]
        
        # make plots
        make_pw_plots(uq,
                    a_test,
                    y_test,
                    n_samples=n_samples,
                    nstd=nstd,
                    k_trunc=k_trunc_pw,
                    FLAG_saveplot=FLAG_SAVE,
                    str_exptype=str_exptype
                    )
        plt.close('all')

# %% Worst, median, best (test)
plt.close('all')
idx_worst = torch.argmax(errors_test, dim=0).item()
idx_median = torch.argsort(errors_test)[errors_test.shape[0]//2, ...].item()
idx_best = torch.argmin(errors_test, dim=0).item()

idxs = [idx_worst, idx_median, idx_best]
names = ["worst_test", "median_test", "best_test"]

# loop over list
if FLAG_PW:
    for loop in range(len(idxs)):
        # Setup data
        ind_test = idxs[loop]
        a_test = input_test[ind_test,:]
        y_test = output_test[ind_test,:]
        
        str_exptype = names[loop]
        
        # make plots
        make_pw_plots(uq,
                    a_test,
                    y_test,
                    n_samples=n_samples,
                    nstd=nstd,
                    k_trunc=k_trunc_pw,
                    FLAG_saveplot=FLAG_SAVE,
                    str_exptype=str_exptype
                    )
        plt.close('all')
