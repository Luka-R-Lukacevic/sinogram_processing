import numpy as np
from ct_object import CT_Object
from tych_radon_mat import lamda_max_function
import ot
from helper_functions import distribution_sample

angle_frequency = 1/6
theta = np.arange(0., 180., 1/angle_frequency)
def f(lamda):
    y, g = lamda_max_function(lamda, theta, twocircles.image.shape, twocircles.sinogram)
    return y, g
mse_values_sci = 0
emd_iradon = 0
y_values = {
    0.3: 0.0,
    1: 0.0,
    1.5: 0.0,
    2: 0.0,
    3: 0.0,
    5: 0.0,
    10: 0.0,
    20: 0.0,
    50: 0.0,
    100: 0.0
}
emd_values_alg = {
    0.3: 0.0,
    1: 0.0,
    1.5: 0.0,
    2: 0.0,
    3: 0.0,
    5: 0.0,
    10: 0.0,
    20: 0.0,
    50: 0.0,
    100: 0.0
}
mse_values_alg = {
    0.3: 0.0,
    1: 0.0,
    1.5: 0.0,
    2: 0.0,
    3: 0.0,
    5: 0.0,
    10: 0.0,
    20: 0.0,
    50: 0.0,
    100: 0.0
}

for i in range(1,11):
    twocircles = CT_Object(np.zeros([25,25]))
    twocircles.add_circ_im(0.5,0.5,0.2,1)
    twocircles.add_circ_im(-0.5,-0.5,0.2,1)
    twocircles.add_noise_to_sinogram(1)
    sinogram = twocircles.sinogram.copy()
    ground_object = twocircles.image.copy()  
    
    mse_values_sci += np.sum(np.square(ground_object - twocircles.recon))
    
    N= 2000
    ground_sample_distr = distribution_sample(ground_object, N)
    iradon_sample_distr = distribution_sample(twocircles.recon, N)
    
    a, b = np.ones((N,)) / N, np.ones((N,)) / N  # uniform distribution on samples
    # loss matrix
    M = ot.dist(ground_sample_distr,iradon_sample_distr)
    emd_iradon += ot.emd2(a, b, M)
    
    
    for x in y_values.keys():
        k,g = f(x)
            
        y_values[x]+=k
        
        mse1 = np.sum(np.square(ground_object - g))
        mse_values_alg[x]+=mse1
        
        lambda_sample_distr = distribution_sample(g, N)
        M = ot.dist(ground_sample_distr,lambda_sample_distr)
        emd_lambda = ot.emd2(a, b, M)
        emd_values_alg[x]+=emd_lambda
        
        #print(f"x: {x}, f(x): {k}, MSE: {mse1}, EMD: {emd_lambda}")
        
        
    print("Turn: ",i)

for x in y_values.keys():
    y_values[x] /=10
    mse_values_alg[x]/=10
    emd_values_alg[x]/=10

print("y_values: ", y_values)
print("mse_values_alg: ", mse_values_alg)
print("mse_values_sci: ",mse_values_sci/10)
print("emd_values_alg: ", emd_values_alg)
print("emd_iradon: ",emd_iradon/10)



