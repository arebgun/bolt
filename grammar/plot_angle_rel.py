import sys
sys.path.insert(1,'..')
import sempoles as sem
import probability_function as pfunc
from matplotlib import pyplot as plt
import numpy as np

new_left_func = pfunc.LogisticBell(loc=-89.776710082, scale=43.3299251684, domain=sem.angle_domain)
new_right_func = pfunc.LogisticBell(loc=95.4602457612, scale=55.6525785799, domain=sem.angle_domain)
new_front_func = pfunc.LogisticBell(loc=1.02103143313, scale=48.3283646849, domain=sem.angle_domain)
new_back_func = pfunc.LogisticBell(loc=-179.113555082, scale=11.5555285078, domain=sem.angle_domain)

new_funcss = [new_left_func,new_right_func,new_front_func,new_back_func]
funcss = [sem.right_func,sem.left_func,sem.front_func,sem.back_func]

theta, r = np.mgrid[0:2*np.pi:360j, 0:1:2j]
deg, r = np.mgrid[0:360:360j, 0:1:2j]
# z = np.random.random(theta.size).reshape(theta.shape)

fig, axess = plt.subplots(nrows=2,ncols=4, subplot_kw=dict(projection='polar'))

for row, funcs, cmap, titles in zip(axess,[funcss,new_funcss],('Oranges','Blues'),(['','','',''],['left','right','front','back'])):
    for ax, func, title in zip(row,funcs,titles):
        z = func(deg)
        ax.pcolormesh(theta, r, z, shading='gouraud',cmap=cmap)
        # ax.subtitle(title)
        ax.set_theta_offset(-np.pi/2.)
        ax.set_xticks(np.pi/180. * np.linspace(180,  -180, 4, endpoint=False))
        ax.set_ylim([0, 1])
        ax.set_yticklabels([])
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=-0.5)
plt.subplots_adjust(hspace=-0.2)
plt.show()