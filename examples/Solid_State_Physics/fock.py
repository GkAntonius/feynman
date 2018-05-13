"""Create the Fock interaction diagram."""

import matplotlib.pyplot as plt
from feynman import Diagram

diagram = Diagram()


v1 = diagram.vertex(xy=(.1,.5), marker='')
v2 = diagram.vertex(xy=(.3,.5))
v3 = diagram.vertex(xy=(.7,.5))
v4 = diagram.vertex(xy=(.9,.5), marker='')

l12 = diagram.line(v1, v2, arrow=True)
w23 = diagram.line(v2, v3, style='elliptic wiggly')
l23 = diagram.line(v2, v3, arrow=True)
l34 = diagram.line(v3, v4, arrow=True)

l12.text("p")
w23.text("q")
l23.text("p-q")
l34.text("p")

diagram.plot()
plt.savefig('pdf/fock.pdf')
diagram.show()

