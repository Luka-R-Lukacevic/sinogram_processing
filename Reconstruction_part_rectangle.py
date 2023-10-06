import numpy as np
from ct_object import CT_Object
from tych_radon_mat import lamda_max_function
import matplotlib.pyplot as plt
import ot
from helper_functions import distribution_sample

angle_frequency = 1/2
theta = np.arange(0., 180., 1/angle_frequency)

rectangle = CT_Object(np.zeros([64,64]))
rectangle.add_rec_im(0,0,1,0.3,0.5)
sinogram = rectangle.sinogram.copy()
ground_object = rectangle.image.copy()

rectangle.plot()
rectangle.plot_sin()
rectangle.plot_center()
rectangle.plot_rec()

def f(lamda):
    y, g = lamda_max_function(lamda, theta, rectangle.image.shape, rectangle.sinogram)
    return y, g


#x_values = np.linspace(0.001, 100, 30)
x_values  = [0.01, 0.1, 0.3, 0.5,1,1.5,2,3,5,10]
y_values = []
emd_values_alg = []
mse_values_alg = []
mse_values_sci = np.sum(np.square(ground_object - rectangle.recon))

N= 10000
ground_sample_distr = distribution_sample(ground_object, N)
iradon_sample_distr = distribution_sample(rectangle.recon, N)

a, b = np.ones((N,)) / N, np.ones((N,)) / N  # uniform distribution on samples
# loss matrix
M = ot.dist(ground_sample_distr,iradon_sample_distr)
emd_iradon = ot.emd2(a, b, M)


for x in x_values:
    k,g = f(x)
        
    y_values.append(k)
    
    mse1 = np.sum(np.square(ground_object - g))
    mse_values_alg.append(mse1)
    
    lambda_sample_distr = distribution_sample(g, N)
    M = ot.dist(ground_sample_distr,lambda_sample_distr)
    emd_lambda = ot.emd2(a, b, M)
    emd_values_alg.append(emd_lambda)
    
    print(f"x: {x}, f(x): {k}, MSE: {mse1}, EMD: {emd_lambda}")
    
    
plt.plot(x_values, y_values)
plt.xlabel("x")
plt.ylabel("y")
plt.show()

plt.plot(x_values,mse_values_alg)
plt.xlabel("x")
plt.ylabel("emd")
plt.show()

print("mse_values_alg: ", mse_values_alg)
print("mse_values_sci: ",mse_values_sci)
print("emd_values_alg: ", emd_values_alg)
print("emd_iradon: ",emd_iradon)
print("f_values_alg: ", y_values)


