import numpy as np
from ct_object import CT_Object


circle = CT_Object(np.zeros([128,128]))
circle.add_circ_im(0.2,0.2,0.25,1)
circle.plot_sin()


circle.pert_sin(0,3)
circle.plot_sin()


circle.plot_rec()


circle.rec_sin()
circle.plot_rec()


circle.plot_sin()