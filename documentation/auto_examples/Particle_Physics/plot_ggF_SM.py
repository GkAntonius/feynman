"""
ggFZZ SM
========

The gluon-gluon fusion.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.,10.))
ax = fig.add_axes([0,0,1,1], frameon=False)

diagram = Diagram(ax)
#diagram.text(.5,0.9,"Gluon-Gluon Fusion (ggF)",fontsize=40)
in1 = diagram.vertex(xy=(.1,.7), marker='')
in2= diagram.vertex(xy=(.1,.3), marker='')
v1 = diagram.vertex(xy=(.4,.7))
v2 = diagram.vertex(xy=(.4,.3))
v3 = diagram.vertex(xy=(.6,.5))
higgsout = diagram.vertex(xy=(.9,.5))

gluon_style = dict(style='linear loopy', xamp=.025, yamp=.035, nloops=7)

g1 = diagram.line(in1, v1, **gluon_style)
g2 = diagram.line(in2, v2, **gluon_style)
t1 = diagram.line(v1, v2)
t2 = diagram.line(v2, v3)
t3 = diagram.line(v3, v1)
higgs = diagram.line(v3, higgsout, arrow=False, style='dashed')

g1.text("g",fontsize=30)
g2.text("g",fontsize=30)
t1.text("t",fontsize=30)
t2.text("t",fontsize=30)
t3.text(r"$\bar{\mathrm{t}}$",fontsize=35)
higgs.text("H",fontsize=30)

diagram.plot()
plt.show()
