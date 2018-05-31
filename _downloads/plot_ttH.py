"""
ttH
===

The ttH diagram.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.,10.))
ax = fig.add_axes([0,0,1,1], frameon=False)

diagram = Diagram(ax)
diagram.text(.5,0.9,"Associated Top Pair (ttH)", fontsize=40)
in1 = diagram.vertex(xy=(.1,.8), marker='')
in2= diagram.vertex(xy=(.1,.2), marker='')
v1 = diagram.vertex(xy=(.4,.7))
v2 = diagram.vertex(xy=(.4,.3))
v3 = diagram.vertex(xy=(.6,.5))
out1 = diagram.vertex(xy=(.9,.8), marker='')
out2 = diagram.vertex(xy=(.9,.2), marker='')
higgsout = diagram.vertex(xy=(.9,.5))

g1 = diagram.line(in1, v1, style='loopy',nloops=7,yamp=0.04)
g2 = diagram.line(in2, v2, style='loopy',nloops=7,yamp=0.04)
t1 = diagram.line(v3, v1, arrow = True)
t2 = diagram.line(v2, v3, arrow = True)
higgs = diagram.line(v3, higgsout, arrow=False, style='dashed')
t3 = diagram.line(v1, out1, arrow=True)
t4 = diagram.line(out2, v2, arrow=True)

g1.text("g",fontsize=30)
g2.text("g",fontsize=30)
diagram.text(v3.xy[0], v3.xy[1]+0.1, r"$\bar{\mathrm{t}}$",fontsize=35)
t2.text("t",fontsize=30)
t3.text("t",fontsize=30)
t4.text(r"$\bar{\mathrm{t}}$",fontsize=30)
higgs.text("H",fontsize=35)

diagram.plot()
plt.show()
