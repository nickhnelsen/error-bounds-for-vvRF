import torch
import math

class GaussianRF(object):
    """       
    Samples N GRFs with Matern covariance with periodic BCs only; supports spatial dimensions up to d=3.
    Requires for PyTorch version >=1.8.0 (torch.fft).
    """
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic", device=None):
        """
        Size must be even.
        """
        
        if size % 2 != 0:
            print("ERROR: ``size'' must be even.")

        self.dim = dim
        self.device = device

        if sigma is None:
            sigma = tau**(0.5*(2*alpha - self.dim))

        k_max = size//2

        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                           torch.arange(start=-k_max, end=0, step=1, device=device)), 0)

            self.sqrt_eig = size*sigma*((4*(math.pi**2)*(k**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0] = 0.0

        elif dim == 2:
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,1)

            k_x = wavenumbers.transpose(0,1)
            k_y = wavenumbers

            self.sqrt_eig = (size**2)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0] = 0.0

        elif dim == 3:
            wavenumbers = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device), \
                                    torch.arange(start=-k_max, end=0, step=1, device=device)), 0).repeat(size,size,1)

            k_x = wavenumbers.transpose(1,2)
            k_y = wavenumbers
            k_z = wavenumbers.transpose(0,2)

            self.sqrt_eig = (size**3)*sigma*((4*(math.pi**2)*(k_x**2 + k_y**2 + k_z**2) + tau**2)**(-alpha/2.0))
            self.sqrt_eig[0,0,0] = 0.0

        self.size = []
        for j in range(self.dim):
            self.size.append(size)

        self.size = tuple(self.size)

    def sample(self, N):
        """
        Input:
            N: (int), number of GRF samples to return
        
        Output:
            u: (N, size, ..., size) tensor
        """

        coeff = torch.randn(math.ceil(N/2), *self.size, 2, device=self.device) # real and imag iid N(0,1)
        coeff = self.sqrt_eig*(coeff[...,0] + coeff[...,1]*1.j) # complex KL expansion coefficients
        
        u = torch.fft.ifftn(coeff, s=self.size)     # u_1 + u_2 i
        u = torch.cat((torch.real(u), torch.imag(u)), 0)
        
        if N % 2 == 0:
            return u
        else: # N is odd, drop one sample before returning
            return u[:-1,...] 
