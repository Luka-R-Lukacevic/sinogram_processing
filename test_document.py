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



rec = CT_Object(np.zeros([256,256]))
rec.add_rec_im(0,0,5,0.3,0.5)
rec.plot()

rec.plot_sin()
rec.pert_sin(0,5)
rec.plot_sin()
rec.plot_rec()
rec.rec_sin()
rec.plot_sin()
rec.plot_rec()


twocircles = CT_Object(np.zeros([256,256]))
twocircles.add_circ_im(0.5,0.5,0.2,1)
twocircles.add_circ_im(-0.5,-0.5,0.2,3)
twocircles.plot()
twocircles.plot_sin()
twocircles.pert_sin(0,5)
twocircles.plot_sin()
twocircles.plot_rec()
twocircles.rec_sin()
twocircles.plot_sin()
twocircles.plot_rec()