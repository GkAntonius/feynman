
import matplotlib.pyplot as plt

from feynman import Diagram

# Set the ax
fig = plt.figure(figsize=(12,3))
ax = fig.add_subplot(111, frameon=False)

ax.set_xlim(0, 3)
ax.set_ylim(0, .75)

y0 = .75 / 2

opwidth = 0.3
linlen = 0.4

W_style = dict(style='double wiggly', nwiggles=2)
v_style = dict(style='simple wiggly', nwiggles=2)

# First diagram
D1 = Diagram(ax)

xy = [0.2, y0]
v01 = D1.verticle(xy)

xy[0] += linlen
v02 = D1.verticle(v01.xy, dx=linlen)

W = D1.line(v01, v02, **W_style)

text_prop = dict(y=0.06, fontsize=22)

W.text("$W$", **text_prop)

D1.text(.75, y0, "=", fontsize=30)

xy = [0.9, y0]
v11 = D1.verticle(xy)
v13 = D1.verticle(v11.xy, dx=opwidth)
v14 = D1.verticle(v13.xy, dx=linlen)

O = D1.operator([v11,v13], c=1.1)
O.text("${\\varepsilon^{-1}}$", x=.0, y=.01, fontsize=35)
D1.line(v13, v14, **v_style)

D1.plot()

fig.savefig('pdf/gw-W.pdf')

