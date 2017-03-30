
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
v01 = D.verticle(xy)
v02 = D.verticle(v01.xy, dx=opwidth)
P = D.operator([v01,v02], c=1.3)
P.text("$P$")

D.text(.70, y0, "=", fontsize=30)

xy[0] = 0.9
v21 = D.verticle(xy)
v22 = D.verticle(xy, dx=linlen)

l21 = D.line(v22, v21, **G_style)
l21 = D.line(v21, v22, **G_style)

D.plot()
fig.savefig('pdf/gw-P.pdf')

