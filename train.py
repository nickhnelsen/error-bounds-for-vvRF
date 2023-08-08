import torch
import numpy as np
import time

import os, sys
from utilities_module import DataReader
from RFM import RandomFeatureModel

def make_save_path(test_str, mypth):
    '''
    Create path to output directory
    Example of input:
        test_str = 'my_exp_num23_parametersHERE'
        mypth = './results/'
    '''
    save_path = mypth + "vvRF_" + test_str +"/"
    return save_path

# %% Problem Setup

# Process command line arguments
print(sys.argv)
m = int(sys.argv[1])                    # number of features
n = int(sys.argv[2])                    # training sample size
J = int(sys.argv[3])                    # data resolution
idx_MC = int(sys.argv[4])               # Monte Carlo/experiment run number
lamreg = float(sys.argv[5])             # regularization strength as in vvRF '23 paper
my_path = sys.argv[6]                   # directory to save output

# Logistic
# data_path = "data/burgers_data_R10.mat"                     # path to dataset
# data_path = '/media/nnelsen/SharedHDD2TB/datasets/error-bounds-for-vvRF/data/burgers_data_R10.mat'
data_path = '/groups/astuart/nnelsen/data/burgers/zl_data_burg/burgers_data_R10.mat'
TEST_STR = "idx" + str(idx_MC) + "_J" + str(J) + "m" + str(m) + "n" + str(n)
user_comment = "vvRF_paper_runs_HPC"
FLAG_SAVE = True
newseed = None              # e.g.: None or int(datetime.today().strftime('%Y%m%d'))

# Problem
lamreg = n * lamreg
K = J                     # resolution, must be power of two
ntest = 500                 # testing sample size (maximum is 2048 - n)
bsize_train = 25
bsize_test = 50
bsize_grf_train = 50
bsize_grf_test = 100
bsize_grf_sample = 500
kmax = 64                   # zero pad after kmax RF GRF modes, kmax <= K//2
var_noise = 0

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

save_path = make_save_path(TEST_STR, my_path)
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

# extract test
input_test = input_train[-ntest:, ...]
input_train = input_train[:-ntest, ...]
output_test = output_train[-ntest:, ...]
output_train = output_train[:-ntest, ...]
    
# shuffle
dataset_shuffle_idx = torch.randperm(input_train.shape[0])
input_train = input_train[dataset_shuffle_idx, ...]
output_train = output_train[dataset_shuffle_idx, ...]

# extract train
input_train = input_train[:n, ...]
output_train = output_train[:n, ...]

# %% Setup model and save hyperparameters

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", dev)

if newseed is not None:
    np.random.seed(newseed)
    torch.manual_seed(newseed)
            
rfm = RandomFeatureModel(K, n, m, ntest, lamreg, nu_rf, al_rf, bsize_train=bsize_train, bsize_test=bsize_test, bsize_grf_train=bsize_grf_train, bsize_grf_test=bsize_grf_test, bsize_grf_sample=bsize_grf_sample, device=dev, al_g=al_g, tau_g=tau_g, sig_g=sig_g, kmax=kmax, sig_rf=sig_rf, K_fine=K_fine)

rfm.load_train(input_train, output_train)
rfm.load_test(input_test, output_test)
rfm.output_train_noisy = output_train

hyp_array = np.asarray((user_comment,newseed,K,m,n,ntest,bsize_train,bsize_test,bsize_grf_train,bsize_grf_test,bsize_grf_sample,kmax,sig_rf,nu_rf,al_rf,sig_g,tau_g,al_g,lamreg,dataset_shuffle_idx,var_noise), dtype=object) # array of hyperparameters

# Write hyperparameters to file
if FLAG_SAVE:
    np.save(save_path + "hyperparameters.npy", hyp_array)

# %% Least squares train and regularization grid sweep

start = time.time() 
rfm.fit()
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

# %% Regsweep

if var_noise == 0:
    e_reg = rfm.regsweep([1e-1, 5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5, 5e-6, 1e-6])
    # e_reg = rfm.regsweep([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12])
print('\n RKHS Norm of sweeped coeff:', torch.linalg.norm(rfm.al_model.cpu()).item()/np.sqrt(rfm.m),'; Max sweeped coeff:', torch.max(torch.abs(rfm.al_model.cpu())).item())

# %% Save to file

if FLAG_SAVE:
    np.save(save_path + 'errors_train_r_b.npy', np.asarray([e_train, b_train]))
    np.save(save_path + 'errors_train.npy', errors_train.cpu().numpy())
    
    np.save(save_path + 'errors_test_r_b.npy', np.asarray([e_test, b_test]))
    np.save(save_path + 'errors_test.npy', errors_test.cpu().numpy())
    
    np.save(save_path + 'errors_reg.npy', e_reg) # lamreg, e_train, b_train, e_test, b_test
    
    np.save(save_path + 'grf_g.npy', rfm.grf_g.cpu().numpy())
    np.save(save_path + 'al_model.npy', rfm.al_model.cpu().numpy())
