from feynman import Diagram
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8.0,2.0))
ax = fig.add_axes([0,0,1,1], frameon=False)

fig.patch.set_alpha(0.)
ax.patch.set_alpha(0.)

ax.set_xlim(0, fig.get_size_inches()[0])
ax.set_ylim(0, fig.get_size_inches()[1])

# Init D and ax
D = Diagram(ax)

D.x0 = 0.2
D.y0 = sum(D.ax.get_ylim()) * .35

# Various size
opwidth = 1.
linlen = 2.
objspace = .8
wiggle_amplitude=.1

# Line styles
Ph_style = dict(style='elliptic loopy', ellipse_spread=.6, xamp=.10, yamp=-.15, nloops=15)
DW_style = dict(style='circular loopy', circle_radius=.7, xamp=.10, yamp=.15, nloops=18)
G_style = dict(style='simple', arrow=True, arrow_param={'width':0.15, 'length': .3})

# Item 1
v1 = D.verticle([D.x0, D.y0])
v2 = D.verticle(v1.xy, dx=opwidth)
Sigma = D.operator([v1,v2])
Sigma.text("$\Sigma^{ep}$")

# Item 2
D.text(v2.x + objspace, D.y0, "=", fontsize=30)

# Item 3
v1 = D.verticle([v2.x + 2 * objspace,  D.y0 - 0.3])
v2 = D.verticle(v1.xy, dx=linlen)
G = D.line(v1, v2, **G_style)
Ph = D.line(v1, v2, **Ph_style)

# Item 2
D.text(v2.x + objspace, D.y0, "+", fontsize=30)

# Item 3
v1 = D.verticle([v2.x + 3 * objspace,  D.y0 - 0.3])
DW = D.line(v1, v1, **DW_style)

D.plot()

fig.savefig('pdf/phonons-Sigma.pdf')
