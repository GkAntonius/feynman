"""
ggF EFT
=========

The gluon-gluon fusion.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.,10.))
ax = fig.add_axes([0,0,1,1], frameon=False)

diagram = Diagram(ax)
in1 = diagram.vertex(xy=(.1,.6), marker='')
in2= diagram.vertex(xy=(.1,.4), marker='')
v1 = diagram.vertex(xy=(.4,.6))
v2 = diagram.vertex(xy=(.4,.4))
v3 = diagram.vertex(xy=(.6,.5))
v4 = diagram.vertex(xy=(.34,.5), marker='')
higgsout = diagram.vertex(xy=(.9,.5))
epsilon = diagram.operator([v4,v3], c=1.1)
epsilon.text("Effective \n coupling", fontsize=30)

gluon_up_style = dict(style='linear loopy', xamp=.025, yamp=.035, nloops=7)
gluon_down_style = dict(style='linear loopy', xamp=.025, yamp=-.035, nloops=7)

g1 = diagram.line(in1, v1, **gluon_up_style)
g2 = diagram.line(in2, v2, **gluon_down_style)

higgs = diagram.line(v3, higgsout, arrow=False, style='dashed')

g1.text("g",fontsize=30)
diagram.text(v4.xy[0]-.08, v4.xy[1]-.05, "g",fontsize=35)
higgs.text("H",fontsize=30)

diagram.plot()
plt.show()
