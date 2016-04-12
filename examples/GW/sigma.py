
import matplotlib.pyplot as plt

from feynman import Diagram

# Set the ax
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

for spine in ax.spines.values():
    spine.set_visible(False)


ax.set_xlim(0, 2)
ax.set_ylim(0, 1.5)

y0 = sum(ax.get_ylim()) / 2

opwidth = 0.3
linlen = 0.8
tail_marker = 'o'
Gamma_width = .3

W_style = dict(style='double wiggly elliptic', nwiggles=5)
G_style = dict(style='double', arrow=True, arrow_param={'width':0.05})

# First diagram
D1 = Diagram(ax)

arrow_param = dict(width=0.05)

xy = [0.2, y0]
v01 = D1.verticle(xy)
xy[0] += opwidth
v02 = D1.verticle(xy)
Sigma = D1.operator([v01,v02])
Sigma.text("$\Sigma$")

D1.text(.70, y0, "=", fontsize=30)

xy[1] = y0 - 0.05

xy[0] = 0.9
v21 = D1.verticle(xy)
xy[0] += linlen
v22 = D1.verticle(xy)
l21 = D1.line(v21, v22, **G_style)
l22 = D1.line(v21, v22, **W_style)

l21.text("G",y=.05)
l22.text("W", y=.07)

D1.plot()
fig.savefig('Sigma-GW.pdf')

