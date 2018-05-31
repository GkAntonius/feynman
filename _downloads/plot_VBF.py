"""
VBH
===

Vector Boson Fusion.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(10.,10.))
ax = fig.add_axes([0,0,1,1], frameon=False)

diagram = Diagram(ax)
diagram.text(.5,0.9,"Vector Boson Fusion (VBF)",fontsize=40)
in1 = diagram.vertex(xy=(.1,.8), marker='')
in2= diagram.vertex(xy=(.1,.2), marker='')
v1 = diagram.vertex(xy=(.4,.7))
v2 = diagram.vertex(xy=(.4,.3))
v3 = diagram.vertex(xy=(.6,.5))
out1 = diagram.vertex(xy=(.9,.8), marker='')
out2 = diagram.vertex(xy=(.9,.2), marker='')
higgsout = diagram.vertex(xy=(.9,.5))

q1 = diagram.line(in1, v1, arrow=False)
q2 = diagram.line(in2, v2, arrow=False)
wz1 = diagram.line(v1, v3, style='wiggly')
wz2 = diagram.line(v2, v3, style='wiggly')
higgs = diagram.line(v3, higgsout, style='dashed', arrow=False)
q3 = diagram.line(v1, out1, arrow=False)
q4 = diagram.line(v2, out2, arrow=False)

q1.text("$q_1$",fontsize=30)
q2.text("$q_2$",fontsize=30)
diagram.text(v3.xy[0], v3.xy[1]+0.11, "$Z/W^\pm$",fontsize=35)
wz2.text("$Z/W^\pm$",fontsize=35)
q3.text("$q_3$",fontsize=30)
q4.text("$q_4$",fontsize=30)
higgs.text("$H$",fontsize=30)

diagram.plot()
plt.show()
