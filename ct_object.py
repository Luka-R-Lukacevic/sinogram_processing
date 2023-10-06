import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal
from skimage.transform import radon, iradon
from curve_fitting import fit_curve
from helper_functions import shift_array
from helper_functions import center

angle_frequency = 1/2
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
        
        #Now let's define the center shape function of a circle
        self.curve_center = np.zeros(len(theta))
        self.update_curve_center()
        
    
    def update_curve_center(self):    
        for thet in theta:
            thet = int(angle_frequency*thet)
            self.curve_center[thet] = center(self.sinogram[:,thet])

        
    #add a circle or ellipse to the bare image
    def add_circ_im(self,x,y,r,f,a=1,b=1):
        self.image[np.sqrt(((self.xv-x)/a)**2+((self.yv-y)/b)**2)<r] = f
        
        #update sinogram and reconstruction
        self.sinogram = radon(self.image, theta=theta)
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')
        
        #update
        self.update_curve_center()
        
    #plot the image
    def plot(self):
        plt.imshow(self.image, origin='lower')
        plt.axis('off')
        plt.colorbar()
        plt.show()
    
    def plot_center(self):
        plt.plot(self.curve_center)
        plt.ylim(0, 100)
        plt.xlabel('Angle in degrees')
        plt.ylabel('Center')
        plt.show()
    
    
    #plot the sinogram
    def plot_sin(self):
        plt.imshow(self.sinogram, origin='lower')
        plt.axis('off')
        plt.colorbar()
        plt.show()
    
    #plot the reconstructed image
    def plot_rec(self):
        plt.imshow(self.recon, origin='lower')
        plt.axis('off')
        plt.colorbar()
        plt.show()
            
    #add noise/pertubation to the sinogram
    def pert_sin(self,mu,sig2):
        for thet in theta:
            #w will be a normal i.i.d. additive noise, we then shift the whole column up or down by this amount
            w = int(normal(loc=mu, scale=sig2, size=1))
            self.sinogram[:,int(angle_frequency*thet)] = shift_array(self.sinogram[:,int(angle_frequency*thet)], w)
        #update
        self.update_curve_center()        
        #update the reconstruction
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')
    
    def add_noise_to_sinogram(self, sigma2):
        noise_pic = np.random.normal(0, sigma2, size=self.sinogram.shape)
        self.sinogram = self.sinogram + noise_pic
        #update
        self.update_curve_center()
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
        self.update_curve_center()

    
    #recover the sinogram
    def rec_sin(self, func = "poly", reconstruction_type = "orthogonal", recon_number = 2):
        a = np.zeros(len(theta))
        b = np.zeros(len(theta))
        
        if reconstruction_type == "orthogonal":
            a = fit_curve(self.curve_center,func)
            for t in theta:
                b[int(angle_frequency*t)] = center(self.sinogram[:,int(angle_frequency*t)])
                #a[int(angle_frequency*t)] = round(a[int(angle_frequency*t)])
        e = a-b
                        
        def correct_the_jumps(arr):
            indices = np.where(np.abs(np.diff(arr)) >= 0.5)[0] + 1
            indices = np.concatenate([indices, [len(arr)-1]])
            for j in range(indices[0]):
                arr[j] = round(arr[j])
            for i in range(len(indices)-1):
                if arr[indices[i]] > arr[indices[i]-1]:
                    for j in range(indices[i], indices[i+1]):
                        arr[j] = arr[indices[i]-1] + 1
                else:
                    for j in range(indices[i], indices[i+1]):
                        arr[j] = arr[indices[i]-1] - 1           
            #print(arr)
            return arr
        f = e.copy()
        f = correct_the_jumps(f)
        
        for thet in theta:
            #print(round(e[int(angle_frequency*thet)]),e[int(angle_frequency*thet)],f[int(angle_frequency*thet)])
            if recon_number == 2:
                self.sinogram[:,int(angle_frequency*thet)] = shift_array(self.sinogram[:,int(angle_frequency*thet)], round(f[int(angle_frequency*thet)]))
            else:
                self.sinogram[:,int(angle_frequency*thet)] = shift_array(self.sinogram[:,int(angle_frequency*thet)], round(e[int(angle_frequency*thet)]))
        #update
        self.update_curve_center()
        #print("finished")
        self.recon = iradon(self.sinogram, theta=theta, filter_name='ramp')
        '''
        for k in range(len(self.recon)):
            for l in range(len(self.recon)):
                self.recon[k,l] = int(round(self.recon[k,l],0))
        '''