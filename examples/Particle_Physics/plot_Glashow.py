"""
Glashow resonance
=================

Example of drag and droppable labels.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.,10.))
ax = fig.add_axes([0,0,1,1], frameon=False)

diagram = Diagram(ax, draggable=True)
i1 = diagram.vertex(xy=(.15, .8), marker="")
i2 = diagram.vertex(xy=(.15, .2), marker="")
v1 = diagram.vertex(xy=(.35, .5))
v2 = diagram.vertex(xy=(.65, .5))
o1 = diagram.vertex(xy=(.85, .8), marker="")
o2 = diagram.vertex(xy=(.85, .2), marker="")

f1 = diagram.line(i1, v1, arrow = True)
f2 = diagram.line(i2, v1, arrow = True)
w1 = diagram.line(v1, v2, style = "wiggly")
f3 = diagram.line(v2, o1, arrow = True)
f4 = diagram.line(v2, o2, arrow = True)

opts = {"fontsize": 30}
i1.text(r"$\bar{\nu}_e (p_1)$",    .003,  .035, **opts)
i2.text(r"$e^- (p_2)$",           -.005, -.037, **opts)
o1.text(r"$\bar{\nu}_\mu (p_3)$",  .006,  .035, **opts)
o2.text(r"$\mu^- (p_4)$",          .024, -.032, **opts)
w1.text(r"$W^- (k)$",              .340,  .054, **opts)

diagram.plot()
plt.show()

# If text element have been dragged, then their relative coordinates are
# updated accordingly. Let us print those.
for drawable in (i1, i2, o1, o2, w1):
    s, x, y, _ = drawable.texts[0]
    print("{:24s} {:6.3f} {:6.3f}".format(s, x, y))
