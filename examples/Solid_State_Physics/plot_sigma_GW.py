"""
GW self-energy
==============

A diagram containing double lines.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(8,2))
ax = fig.add_axes([0,0,1,1], frameon=False)

ax.set_xlim(0, 1)
ax.set_ylim(0,.25)

# Sigma operator
D = Diagram(ax)

v11 = D.vertex([.1, .08])
v12 = D.vertex(v11.xy, dx=.15)

Sigma = D.operator([v11, v12])
Sigma.text("$\Sigma$")

# Symbols
D.text(v12.x+.1, v12.y, "=")

# GW convolution
v21 = D.vertex(v12.xy, dxy=[0.2, -0.04])
v22 = D.vertex(v21.xy, dx=0.4)

l21 = D.line(v21, v22, style='double', arrow=True)

# Specifying the number of wiggles and the amplitude of the wiggles
l22 = D.line(v21, v22, style='double wiggly elliptic', nwiggles=5.5, amplitude=0.015)

# 't' is the coordinate along the line (from 0 to 1)
l21.text("G", t=0.4, y=.025, fontsize=24)
l22.text("W", y=-.06, fontsize=24)

# Plot and show
D.plot()
plt.show()
