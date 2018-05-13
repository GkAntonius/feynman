
import numpy as np
import matplotlib.pyplot as plt

from feynman import Diagram

# Set the ax
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, frameon=False)

ax.set_xlim(0, 2)
ax.set_ylim(0, 1.5)

y0 = sum(ax.get_ylim()) / 2

opwidth = 0.3
linlen = 0.8
tail_marker = 'o'
Gamma_width = .3

W_style = dict(style='double wiggly', nwiggles=4)
G_style = dict(style='double elliptic',
                ellipse_excentricity=-1.2, ellipse_spread=.3,
                arrow=True, arrow_param={'width':0.05})

D = Diagram(ax)

xy = [0.2, y0]
v01 = D.vertex(xy)
v02 = D.vertex(v01.xy, dx=opwidth)
P = D.operator([v01,v02], c=1.3)
P.text("$P$")

D.text(.70, y0, "=", fontsize=30)

xy[0] = 0.9
v21 = D.vertex(xy)

xy[0] += linlen
xy[1] += Gamma_width / 2
v22 = D.vertex(xy)

xy[1] += - Gamma_width
v23 = D.vertex(xy)

xy[1] +=  Gamma_width / 2
xy[0] +=  Gamma_width * np.sqrt(3) / 2 
v24 = D.vertex(xy)

l21 = D.line(v22, v21, **G_style)
l21 = D.line(v21, v23, **G_style)

Gamma = D.operator([v22,v23, v24])
Gamma.text("$\Gamma$")

D.plot()

fig.savefig('pdf/hedin-P.pdf')

