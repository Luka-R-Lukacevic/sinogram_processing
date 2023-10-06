import numpy as np
from ct_object import CT_Object
import ot
import ot.plot
from helper_functions import distribution_sample

angle_frequency = 1
theta = np.arange(0., 180., 1/angle_frequency)

mse_recon = []
mse_per = []
emd_recon = []
emd_per = []

for j in range(30):
    rectangle = CT_Object(np.zeros([128,128]))
    rectangle.add_rec_im(0,0,1,0.3,0.5)
    ground_sinogram = rectangle.sinogram.copy()
    
    
    
    rectangle.plot()
    rectangle.plot_sin()
    
    
    rectangle.pert_sin(0,5)
    rectangle.plot_center()
    pertubated_sinogram = rectangle.sinogram.copy()
    
    
    rectangle.plot_sin()
    rectangle.plot_rec()
    
    rectangle.rec_sin("poly","orthogonal",1)
    rectangle.rec_sin("poly","orthogonal",1)
    rectangle.rec_sin("poly","orthogonal",2)
    reconstructed_sinogram = rectangle.sinogram.copy()
    
    
    rectangle.plot_sin()
    rectangle.plot_rec()
    
    
    N= 2000
    ground_sample_distr = distribution_sample(ground_sinogram, N)
    pertubated_sample_distr = distribution_sample(pertubated_sinogram, N)
    reconstructed_sample_distr = distribution_sample(reconstructed_sinogram, N)
    
    a, b = np.ones((N,)) / N, np.ones((N,)) / N  # uniform distribution on samples
    # loss matrix
    M1 = ot.dist(ground_sample_distr,reconstructed_sample_distr)
    M2 = ot.dist(ground_sample_distr,pertubated_sample_distr)
    emd_processed1 = ot.emd2(a, b, M1)
    emd_processed2 = ot.emd2(a, b, M2)
    print("EMD reconstructed: ", emd_processed1)
    print("EMD noisy: ", emd_processed2)
    emd_recon.append(emd_processed1)
    emd_per.append(emd_processed2)
    
    
    mse1 = np.sum(np.square(ground_sinogram - reconstructed_sinogram))
    mse2 = np.sum(np.square(ground_sinogram - pertubated_sinogram))
    print("Mean Squared Error reconstructed: ", mse1)
    print("Mean Squared Error noisy: ", mse2)
    
    mse_recon.append(mse1)
    mse_per.append(mse2)

emd_recon = np.array(emd_recon)
emd_per = np.array(emd_per)

print("Final EMD reconstructed: ", np.mean(emd_recon))
print("Final EMD noisy: ", np.mean(emd_per))



mse_recon = np.array(mse_recon)
mse_per = np.array(mse_per)

print("Final Mean Squared Error reconstructed: ", np.mean(mse_recon))
print("Final Mean Squared Error noisy: ", np.mean(mse_per))
