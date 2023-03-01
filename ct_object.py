import numpy as np
import skimage
import matplotlib.pyplot as plt


from numpy.random import normal
from skimage.transform import radon, iradon

from curve_fitting import fit_curve


from helper_functions import highest_non_zero_index
from helper_functions import lowest_non_zero_index
from helper_functions import wasserstein_distance
from helper_functions import shift_array


angle_frequency = 2 
theta = np.arange(0., 180., 1/angle_frequency)

class CT_Object ():
    
    def __init__(self,image):
        #Define the image
        self.image = image
        
        #xv and yv can be used to describe geometric objects
        _ = np.linspace(-1, 1, self.image.shape[0])
        self.xv, self.yv = np.meshgrid(_,_)
        
        #Define the sinogram
        self.sinogram = radon(self.image, theta=theta)
        
        #Define the reconstructed image
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')
        
        #Now let's define the upper shape function of a circle
        
        self.curve_up = np.zeros(len(theta))
        self.update_curve_up()
        self.curve_down = np.zeros(len(theta))
        self.update_curve_down()
        
    def update_curve_up(self):    
        for thet in theta:
            thet = int(angle_frequency*thet)
            self.curve_up[thet] = highest_non_zero_index(self.sinogram[:,thet])

    def update_curve_down(self):    
        for thet in theta:
            thet = int(angle_frequency*thet)
            self.curve_down[thet] = lowest_non_zero_index(self.sinogram[:,thet])

        
    #add a circle or ellipse to the bare image
    def add_circ_im(self,x,y,r,f,a=1,b=1):
        self.image[((self.xv-x)/a)**2+((self.yv-y)/b)**2<r**2] = f
        
        #update sinogram and reconstruction
        self.sinogram = radon(self.image, theta=theta)
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')
        
        #filter low values
        for element in np.nditer(self.recon):
            element = int(element)
        
        #update
        self.update_curve_up()
        self.update_curve_down()
        
    #plot the image
    def plot(self):
        plt.imshow(self.image)
        plt.show()
    
    #plot the sinogram
    def plot_sin(self):
        plt.imshow(self.sinogram)
        plt.show()
    
    #plot the reconstructed image
    def plot_rec(self):
        plt.imshow(self.recon)
        plt.show()
            
    #add noise/pertubation to the sinogram
    def pert_sin(self,mu,sig2):
        for thet in theta:
                #w will be a normal i.i.d. additive noise, we then shift the whole column up or down by this amount
                w = int(normal(loc=mu, scale=sig2, size=1))
                self.sinogram[:,int(2*thet)] = shift_array(self.sinogram[:,int(2*thet)], w)

        #update the reconstruction
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')
    
    
    
    #add a rectangle to the bare image
    def add_rec_im(self,x,y,f,a,b):
        self.image[np.logical_and(np.absolute(self.xv-x)<a, np.absolute(self.yv-y)<b)] = f
        
        #update sinogram and reconstruction
        self.sinogram = radon(self.image, theta=theta)
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')
        
        #filter low values
        for element in np.nditer(self.recon):
            element = int(element)
        
        #update
        self.update_curve_up()
        self.update_curve_down()


    
    
    #recover the sinogram
    def rec_sin(self, func = "sin", reconstruction_type = "up"):
        a = np.zeros(len(theta))
        b = np.zeros(len(theta))
        if reconstruction_type == "up":
            a = fit_curve(self.curve_up,func)
            for t in theta:
                b[int(2*t)] = highest_non_zero_index(self.sinogram[:,int(2*t)])
        
        if reconstruction_type == "down":
            a = fit_curve(self.curve_down,func)
            for t in theta:
                b[int(2*t)] = lowest_non_zero_index(self.sinogram[:,int(2*t)])

        if reconstruction_type == "both":
            a = fit_curve(self.curve_up,func)
            c = fit_curve(self.curve_down,func)
            d = np.zeros(len(theta))
            for t in theta:
                b[int(2*t)] = highest_non_zero_index(self.sinogram[:,int(2*t)])
                d[int(2*t)] = lowest_non_zero_index(self.sinogram[:,int(2*t)])
            a = a + c
            a = a * (0.5)
            b = b + d
            b = b * (0.5)

        e = a-b
        
        for thet in theta:
            self.sinogram[:,int(2*thet)] = shift_array(self.sinogram[:,int(2*thet)], round(e[int(2*thet)]))
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')