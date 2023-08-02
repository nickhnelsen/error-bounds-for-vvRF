import torch
import torch.nn.functional as F
import math
import numpy as np

from utilities_module import dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)
from timeit import default_timer

class RandomFeatureModel:
    '''
    Class implementation of the random feature model for Burgers' equation solution map.
    '''
    def __init__(self, K=512, 
                 n=1000, 
                 m=1000, 
                 ntest=1000, 
                 lamreg=0., 
                 nu_rf=2.5e-3, 
                 al_rf=4., 
                 bsize_train=None, 
                 bsize_test=None, 
                 bsize_grf_train=None, 
                 bsize_grf_test=None, 
                 bsize_grf_sample=None, 
                 device=None, 
                 al_g=None, 
                 tau_g=None, 
                 sig_g=None, 
                 kmax=None,
                 sig_rf=1.,
                 var_noise=0,
                 K_fine=8192):
        '''
        Arguments:
            K:              (int), number of mesh points in one spatial direction (a power of two)
            
            n:              (int), number of data (max: 1024) (max: 1000 for Zongyi)
            
            m:              (int), number of random features (max: 1024)
            
            ntest:          (int), number of test points (max: 5000) (max: 1048 for Zongyi)
            
            lamreg:         (float), regularization/penalty hyperparameter strength
            
            nu_rf:          (float), scale parameter in RF filter map
            
            al_rf:          (float), decay parameter in RF filter map
                                                
            bsize_train:    (int), batch size for training
            
            bsize_test:     (int), batch size for testing
            
            bsize_grf_train:      (int), batch size for forming RFM tensors
            
            bsize_grf_test:      (int), GRF batch size for testing
            
            bsize_grf_sample:      (int), batch size for sampling GRFs
            
            device:         torch.device("cuda") for GPU or torch.device("cpu") for CPU
                        
            al_g:           (float), regularity of GRF samples in RF map
            
            tau_g:          (float), length scale of GRF samples in RF map
            
            sig_g:          (float), GRF amplitude hyperparameter
            
            kmax:           (int), Fourier mode truncation level, k=1,...,kmax
                        

        Attributes:
            [arguments]:    (various), see ``Arguments'' above for description

            K_fine:         (int), number of high resolution mesh points (a power of two)
                        
            grf_g:          (K, m, 2), precomputed Gaussian random fields 
                        
            n_max:          (int), max number of data (max: 1024)
            
            ntest_max:      (int), max number of test points (max: 5000)
                        
            tau_g:          (float), inverse length scale (default: 15, 7.5)
            
            al_g:           (float), regularity (default: 2, 2)
            
            al_model:       (m,) numpy array, random feature model expansion coefficents/parameters to be learned
        
            AstarA:         (m, m) numpy array, normal equation matrix
            
            AstarY:         (m,), RHS in normal equations
            
        '''
        
        # From input arguments 
        self.K = K
        self.n = n
        self.m = m
        self.ntest = ntest
        self.lamreg = lamreg                                    # hyperparameter
        self.nu_rf = nu_rf                                      # hyperparameter
        self.al_rf = al_rf                                      # hyperparameter
        self.sig_rf = sig_rf                                    # hyperparameter
        self.var_noise = var_noise
        if bsize_train is None:
            self.bsize_train = 50
        else:
            self.bsize_train = bsize_train
        if bsize_test is None:
            self.bsize_test = 50
        else:
            self.bsize_test = bsize_test
        if bsize_grf_train is None:
            self.bsize_grf_train = 20
        else:
            self.bsize_grf_train = bsize_grf_train
        if bsize_grf_test is None:
            self.bsize_grf_test = self.m
        else:
            self.bsize_grf_test = bsize_grf_test
        if bsize_grf_sample is None:
            if self.m <= 256:
                self.bsize_grf_sample = self.m
            else:
                self.bsize_grf_sample = max(1,self.m//16)
        else:
            self.bsize_grf_sample = bsize_grf_sample
        self.device = device
        self.tau_g = tau_g                                  # hyperparameter
        self.al_g = al_g                                    # hyperparameter
        if sig_g is None:
            self.sig_g = self.tau_g**(0.5*(2.*self.al_g - 1.))
        else:
            self.sig_g = sig_g                                  # hyperparameter
        if kmax is None:
            self.kmax = K//2
        elif kmax > K//2:
            self.kmax = K//2
        else:
            self.kmax = kmax
        self.K_fine = K_fine
            
        # Sample GRFs
        self.grf_g = torch.zeros((m, self.kmax), dtype=torch.cfloat, device=device)
        self.grf = GaussianRFcoeff(self.K, kmax=self.kmax, device=device)
        self.resample()
            
        # Non-input argument attributes of RFM
        self.eps = 1e-8     # nugget to prevent positive values from being zero
        self.al_model = torch.zeros(self.m, device=device)
        self.AstarA = 0 # init to zero now
        self.AstarY = 0 # init to zero now
        self.output_train_noisy = 0

        # Make physical grid spacing
        self.h = 1/self.K
        
        # Wavenumbers
        self.kwave = torch.arange(start=0, end=self.K//2 + 1, step=1, device=device) # positive Nyquist, see \
        # https://pytorch.org/docs/1.8.0/fft.html?highlight=fft#torch.fft.fft
    
    def load_train(self, x, y):
        self.input_train = x
        self.output_train = y
        
    def load_test(self, x, y):
        self.input_test = x
        self.output_test = y
    
    def resample(self):
        for grf_batch in torch.split(self.grf_g, self.bsize_grf_sample):
            grf_batch[...] = self.grf.sample(grf_batch.shape[0])
    
    @staticmethod
    def act_filter(r, al_rf):
        """
        Input:
            r: arbitrary dim torch.tensor
        """
        return F.relu(torch.minimum(2*r, torch.pow(0.5 + r, -al_rf)))
    
    @staticmethod
    def squish(r,a,b):
        """ Maps to (a,b) """
        return (b-a)*torch.sigmoid(r) + a
    
    @staticmethod
    def squish_inv(r,a,b):
        """ Inverts ``squish'' function above """
        return -torch.log(((b - a)/(r - a)) - 1)

    def rf_batch(self, a_batch, g_batch):
        """
        Inputs:
            a_batch: (nbatch, K) tensor of input functions, where K is even
            g_batch: (mbatch, self.kmax) complex tensor of iid Fourier coefficients
            K_fine: (int), fine grid resolution the input data is downsampled from.
        Output:
            Returns (nbatch,mbatch,K) array
        Hyperparameters:
            self.sig_rf
            self.nu_rf
            self.al_rf
            self.sig_g
            self.al_g
            self.tau_g
        """
        PI = math.pi
        sqrt_eig = self.sig_g*((4*(PI**2)*(self.kwave[1:self.kmax+1]**2) + self.tau_g**2)**(-self.al_g/2.0))

        # Define filter mapping
        wave_func = RandomFeatureModel.act_filter(torch.abs(self.nu_rf*self.kwave*2*PI), self.al_rf)

        # Convolve via multiplication in Fourier space
        a_batch = torch.fft.rfft(a_batch)
        g_batch = math.sqrt(2)*sqrt_eig*g_batch
        conv = torch.einsum("n...,m...->nm...", a_batch[...,1:self.kmax+1], g_batch)
    
        # Compute features back in physical space
        return self.sig_rf*F.elu(self.K_fine * torch.fft.irfft(wave_func*self.grf.zeropad(conv), n=self.K))

    def fit(self, a_batch=None, y_batch=None):
        '''
        Solve the (regularized) normal equations given the training data (uses symmetry of A*A)
  
        No Output: 
            --this method only updates the class attributes ``al_model, AstarA, AstarY''
        Data: loaded inside function
            input_train: (n, K), Burgers IC
            output_train: (n, K), Burgers' solution at time 1
        '''
        if a_batch is None and y_batch is None:
            input_train, output_train = self.input_train, self.output_train_noisy
            FLAG_PRINT = True
        else:
            input_train, output_train = a_batch, y_batch
            FLAG_PRINT = False
        train_loader = DataLoader(TensorDataset(input_train, output_train), batch_size=min(self.bsize_train, input_train.shape[0]), shuffle=True)
        self.AstarY = torch.zeros(self.m, device=self.device)
        self.AstarA = torch.zeros((self.m, self.m), device=self.device)
        c = 0
        btchc = 0
        t0 = default_timer()
        for a, y in train_loader:
            # Input and Outputs for this batch
            a, y = a.to(self.device), y.to(self.device)
            to_train = a.shape[0]
        
            # Form RF-based tensors
            ccc = 0
            while 1:
                if self.m - ccc <= 0:
                    break
                elif self.m - ccc >= self.bsize_grf_train:
                    AY_gen = self.bsize_grf_train
                else:
                    AY_gen = self.m - ccc
                # A*Y
                RF = self.rf_batch(a, self.grf_g[ccc:(ccc+AY_gen),:]) # size (to_train, AY_gen, K)
                self.AstarY[ccc:(ccc+AY_gen)] += torch.sum(torch.trapz(torch.einsum("nm...,n...->nm...", RF, y), dx=self.h), dim=0)
                
                # A*A
                cc = 0
                while 1: # fill in rows
                    if ccc+1 - cc <= 0:
                        break
                    elif ccc+1 - cc >= self.bsize_grf_train:
                        AA_gen = self.bsize_grf_train
                    else:
                        AA_gen = ccc+1 - cc
                    RFcc = self.rf_batch(a, self.grf_g[cc:(cc+AA_gen),:])
                    self.AstarA[cc:(cc+AA_gen), ccc:(ccc+AY_gen)] += torch.sum(torch.trapz(torch.einsum("ni...,nj...->nij...", RFcc, RF), dx=self.h), dim=0)
                    cc += AA_gen
                for k in range(AY_gen - 1): # fill in extra columns
                    self.AstarA[ccc+1:(ccc+2+k), ccc+1+k] += torch.sum(torch.trapz(torch.einsum("nm...,n...->nm...", RF[:,1:(k+2),:], RF[:,1+k,:]), dx=self.h), dim=0)
                ccc += AY_gen
   
            # Update
            c += to_train
            btchc += 1
            t1 = default_timer()
            if FLAG_PRINT:
                print("(Training) Batch, Samples, Time Elapsed:", (btchc, c, t1-t0))
        self.AstarA = self.AstarA + self.AstarA.T - torch.diag(torch.diag(self.AstarA)) # symmetry
        self.AstarA /= self.m
        
        # Solve linear system
        if self.lamreg == 0:
            self.AstarAnug = self.AstarA
            self.al_model = torch.squeeze(torch.lstsq(self.AstarY[:,None], self.AstarA)[0])
        else:
            self.AstarAnug = self.AstarA + self.lamreg*torch.eye(self.m, device=self.device)
            self.al_model = torch.linalg.solve(self.AstarAnug, self.AstarY)
    
    def predict(self, a):
        '''
        Evaluate random feature model on a given batch of coefficent functions ``a''.
        Inputs:
            a: (nbatch, K) array or (K,) array
        Output:
            Returns (nbatch, K) array or (K,)
        '''
        a = a.to(self.device)
        FLAG = False
        if a.ndim==1:
            FLAG = True
            a = torch.unsqueeze(a, 0) # size (1, K)

        output_tensor = torch.zeros(a.shape, device=self.device)
        grf_batches = torch.split(self.grf_g, self.bsize_grf_test)
        al_batches = torch.split(self.al_model, self.bsize_grf_test)
        for g, al in zip(grf_batches, al_batches):
            output_tensor += torch.einsum("b,nb...->n...", al, self.rf_batch(a, g))
        
        if FLAG:
            output_tensor = torch.squeeze(output_tensor)
        return output_tensor/self.m
    
    def relative_error_test(self, FLAG_TRAIN=False, FLAG_PRINT=True, FLAG_GETERRORS=False):
        '''
        Compute the expected relative error and Bochner error on the test set.
        '''
        if FLAG_TRAIN:
            input_test, output_test = self.input_train, self.output_train
            progress_str = "(Training Set) Batch, Samples, Time Elapsed:"
        else:
            input_test, output_test = self.input_test, self.output_test
            progress_str = "(Testing Set) Batch, Samples, Time Elapsed:"
        test_loader = DataLoader(TensorDatasetID(input_test, output_test), batch_size=self.bsize_test, shuffle=False)
        ip = InnerProduct1D(self.h)
        c = 0
        btch = 0
        er = 0
        boch_num = 0
        boch_den = 0
        n_samples = input_test.shape[0]
        if FLAG_GETERRORS:
            errors = torch.zeros(n_samples)
        t0 = default_timer()
        for a, y, idx in test_loader:
            # Input and Outputs for this batch
            a, y = a.to(self.device), y.to(self.device)
            to_test = a.shape[0]
       
            # Unscaled error for this batch
            resid = torch.abs(y - self.predict(a)) # (to_test, K)
            boch_num_vec = ip.L2(resid, resid)
            boch_den_vec = ip.L2(y, y)
            resid = torch.sqrt(boch_num_vec/boch_den_vec)
            if FLAG_GETERRORS:
                errors[idx] = resid.cpu()
            er += torch.sum(resid)
            boch_num += torch.sum(boch_num_vec)
            boch_den += torch.sum(boch_den_vec)
        
            # Update
            c += to_test
            btch += 1 
            t1 = default_timer()
            if FLAG_PRINT:
                print(progress_str, (btch, c, t1-t0))
        if FLAG_GETERRORS:
            return er.item()/n_samples, math.sqrt(boch_num/boch_den), errors
        else:
            return er.item()/n_samples, math.sqrt(boch_num/boch_den)
    
    def relative_error_train(self, FLAG_PRINT=True, FLAG_GETERRORS=False):
        '''
        Compute the expected relative error and Bochner error on the training set.
        '''
        return self.relative_error_test(FLAG_TRAIN=True,FLAG_PRINT=FLAG_PRINT,FLAG_GETERRORS=FLAG_GETERRORS)

    def regsweep(self, lambda_list=[1e-6, 1e-7, 1e-8, 1e-9, 1e-10]):
        '''
        Regularization hyperparameter sweep. Requires model to be fit first at least once. Updates model parameters to best performing ones.
        Input:
            lambda_list: (list), list of lambda values
        Output:
            er_store : (len(lambda_list), 5) numpy array, error storage
        '''
        if isinstance(self.AstarA, int):
            raise ValueError("ERROR: Model must be trained at least once before calling ``regsweep''. ")
            return None
        
        al_list = [] # initialize list of learned coefficients
        al_list.append(self.al_model.to("cpu"))
        er_store = np.zeros([len(lambda_list)+1, 5]) # lamreg, e_train, b_train, e_test, b_test
        er_store[0, 0] = self.lamreg
        print('Running \lambda =', er_store[0, 0])
        er_store[0, 1:3] = self.relative_error_train()
        er_store[0, 3:] = self.relative_error_test()
        print('Expected relative error (Train, Test):' , (er_store[0, 1], er_store[0, 3]))
        
        for loop in range(len(lambda_list)):
            reg = lambda_list[loop]
            er_store[loop + 1, 0] = reg 
            print('Running \lambda =', reg)
            
            # Solve linear system
            if reg == 0:
                self.al_model = torch.squeeze(torch.lstsq(self.AstarY[:,None], self.AstarA)[0])
            else:
                self.al_model = torch.linalg.solve(self.AstarA + reg*torch.eye(self.m, device=self.device), self.AstarY)
            al_list.append(self.al_model.to("cpu"))

            # Training error
            er_store[loop + 1, 1:3] = self.relative_error_train()
            
            # Test error
            er_store[loop + 1, 3:] = self.relative_error_test()
            
            # Print
            print('Expected relative error (Train, Test):' , (er_store[loop + 1, 1], er_store[loop + 1, 3]))
            
        # Find lambda with smallest test error and update class regularization attribute
        ind_arr = np.argmin(er_store, axis=0)[3] # smallest test index
        self.lamreg = er_store[ind_arr, 0]
        
        # Update model parameter class attribute corresponding to best lambda
        self.al_model = al_list[ind_arr].to(self.device)
        
        return er_store


class InnerProduct1D:
    '''
    Class implementation of L^2 inner products from samples of data on [0,1].
    '''
    def __init__(self, h):
        '''
        Initializes the class. 
        Arguments:
            h:          Mesh size in x direction.

        Parameters:
            h:          (float), Mesh size in x direction.
                
            quad_type:  (str), Type of numerical quadrature chosen.
        '''
        self.h = h
        self.quad_type = 'trapezoid'
            
    def L2(self, F, G):
        '''
        L^2 inner product
        Inputs: F, G are numpy arrays of size (d1, d2, ..., K), where F*G multiplication must be broadcastable
        '''
        return torch.trapz(F*G, dx=self.h)


class GaussianRFcoeff(object):
    """
    Periodic GRF N(0,1) coefficents
    """
    def __init__(self, size, kmax=None, device=None):
        if size % 2 != 0:
            print("ERROR: ``size'' must be even.")
        self.device = device
        self.size = size
        self.kfreq = size//2
        if kmax is None:
            self.kmax = self.kfreq
        elif kmax > self.kfreq:
            self.kmax = self.kfreq
        else:
            self.kmax = kmax

    def sample(self, N):
        """
        Input:
            N: (int), number of GRF coefficient samples to return
        
        Output:
            u: (N, self.kmax) complex tensor, i.e. only positive frequencies to serve as input to irfft using the convention of rfftfreq; Output must be multiplied by sqrt(2) and root eigenvalues of KL expansion
        """
        iid = torch.randn(N, self.kmax, 2, device=self.device)
        return (iid[...,0] - iid[...,1]*1.j)/2 # complex coefficients, positive indices only
    
    def zeropad(self, ctensor):
        """
        Zero pad a batch of Fourier coefficients
        Input:
            iid: (..., self.kmax) complex tensor
        Output:
            (..., self.kfreq + 1) complex tensor
        """
        coeff = torch.zeros(*ctensor.shape[:-1], self.kfreq + 1, dtype=torch.cfloat, device=self.device)
        coeff[...,1:self.kmax+1] = ctensor
        return coeff