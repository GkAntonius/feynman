"""
Fock exchange
=============

A simple diagram composed of vertices and lines.
"""

import matplotlib.pyplot as plt
from feynman import Diagram

# If no Axes is given, a new one is initialized.
diagram = Diagram()

# Create four vertices.
v1 = diagram.vertex(xy=(.1,.5), marker='')
v2 = diagram.vertex(v1.xy, dx=.2)
v3 = diagram.vertex(v2.xy, dx=.4)
v4 = diagram.vertex(v3.xy, dx=.2, marker='')

# Create four lines.
# By default, 'simple' lines have arrows
# and others flavours such as 'wiggly' or 'loopy' don't.
l12 = diagram.line(v1, v2)
l23 = diagram.line(v2, v3)
l34 = diagram.line(v3, v4, arrow=True)
w23 = diagram.line(v2, v3, style='wiggly elliptic')

# Add labels.
l12.text("p")
w23.text("q")
l23.text("p - q")
l34.text("p")

diagram.plot()
plt.show()
