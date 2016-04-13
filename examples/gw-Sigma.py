
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111, frameon=False)

ax.set_xlim(0, 2)
ax.set_ylim(0, .5)

y0 = 0.175

opwidth = 0.3
linlen = 0.8

W_style = dict(style='double wiggly elliptic', nwiggles=5)
G_style = dict(style='double', arrow=True, arrow_param={'width':0.05})

D = Diagram(ax)

xy = [0.2, y0]
v01 = D.verticle(xy)

xy[0] += opwidth
v02 = D.verticle(xy)

Sigma = D.operator([v01,v02])
Sigma.text("$\Sigma$")

D.text(.70, y0, "=", fontsize=30)

xy[1] = y0 - 0.07

xy[0] = 0.9
v21 = D.verticle(xy)

xy[0] += linlen
v22 = D.verticle(xy)

l21 = D.line(v21, v22, **G_style)
l22 = D.line(v21, v22, **W_style)

l21.text("G", y=.05)
l22.text("W", y=-.1)

D.plot()

fig.savefig('pdf/gw-Sigma.pdf')
fig.savefig('pdf/gw-Sigma.png')

