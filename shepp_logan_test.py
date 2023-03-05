"""
Created on Wed Mar  1 14:58:23 2023

@author: lluka
"""

from ct_object import CT_Object
from skimage.data import shepp_logan_phantom



phantom = CT_Object(shepp_logan_phantom())

phantom.plot()

phantom.plot_sin()

phantom.pert_sin(0,10)
phantom.plot_sin()

phantom.plot_rec()

phantom.rec_sin("polynomial", "orthogonal")
phantom.plot_rec()

phantom.plot_sin()