"""
RPA Polarization
================

A diagram containing an operator.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

# Set up the figure and ax
fig = plt.figure(figsize=(8,2))
ax = fig.add_axes([0,0,1,1], frameon=False)

# It is best to keep the xlim/ylim ratio the same as the figure aspect ratio.
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.25)

y0 = sum(ax.get_ylim()) / 2

# Initialize diagram with the ax
D = Diagram(ax)

# Polarizability operator
v11 = D.vertex([0.1, y0])
v12 = D.vertex(v11.xy, dx=0.15)
P = D.operator([v11, v12], c=1.3)  # c is the excentricity of the patch
P.text("$P$")

# Symbols
D.text(.35, y0, "=", fontsize=30)

# Propagator lines
G_style = dict(style='double elliptic',
               ellipse_excentricity=-1.2, ellipse_spread=.3,
               arrow=True, arrow_param={'width':0.03})

v21 = D.vertex([0.45, y0])
v22 = D.vertex(v21.xy, dx=0.4)

G1 = D.line(v22, v21, **G_style)
G2 = D.line(v21, v22, **G_style)

# Plot and show
D.plot()
plt.show()
