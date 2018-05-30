"""
Electron-phonon coupling self-energy
====================================

A diagram containing loopy lines.
"""
from feynman import Diagram
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,2))
ax = fig.add_axes([0,0,1,1], frameon=False)

ax.set_xlim(0, fig.get_size_inches()[0])
ax.set_ylim(0, fig.get_size_inches()[1])

# Init D and ax
D = Diagram(ax)

D.x0 = 0.2
D.y0 = sum(D.ax.get_ylim()) * .35

# Various size
opwidth = 1.
linlen = 2.
txtpad = .8
wiggle_amplitude=.1

# Line styles
Ph_style = dict(style='elliptic loopy', ellipse_spread=.6, xamp=.10, yamp=-.15, nloops=15)
DW_style = dict(style='circular loopy', circle_radius=.7, xamp=.10, yamp=.15, nloops=18)
G_style = dict(style='simple', arrow=True, arrow_param={'width':0.15, 'length': .3})

# Item 1
v11 = D.vertex([D.x0, D.y0])
v12 = D.vertex(v11.xy, dx=opwidth)
Sigma = D.operator([v11, v12])
Sigma.text("$\Sigma^{ep}$")

# Symbol
D.text(v12.x + txtpad, D.y0, "=")

# Item 3
v21 = D.vertex([v12.x + 2 * txtpad,  D.y0 - 0.3])
v22 = D.vertex(v21.xy, dx=linlen)
G = D.line(v21, v22, **G_style)
Ph = D.line(v21, v22, **Ph_style)

# Symbol
D.text(v22.x + txtpad, D.y0, "+")

# Item 3
v31 = D.vertex([v22.x + 3 * txtpad,  D.y0 - 0.3])
DW = D.line(v31, v31, **DW_style)

D.plot()
plt.show()
