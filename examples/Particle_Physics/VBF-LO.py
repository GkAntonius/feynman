import matplotlib

from feynman import Diagram

fig = matplotlib.pyplot.figure(figsize=(1.,1.))
ax = fig.add_axes([0,0,10,10], frameon=False)
diagram = Diagram(ax)
in1 = diagram.vertex(xy=(.1,.8), marker='')
in2= diagram.vertex(xy=(.1,.2), marker='')
v1 = diagram.vertex(xy=(.5,.7))
v2 = diagram.vertex(xy=(.5,.3))
v3 = diagram.vertex(xy=(.5,.5))
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

q1.text(r"$\bar{q}$",fontsize=30)
q2.text("$Q$",fontsize=30)
diagram.text(v3.xy[0]+0.12, v3.xy[1]+0.11, "$Z/W^\pm$",fontsize=30)
wz2.text("$Z/W^\pm$",fontsize=30)
q3.text(r"$\bar{q}$",fontsize=30)
q4.text("$Q$",fontsize=30)
higgsout.text("$H$",fontsize=30)

diagram.text(v1.xy[0], v1.xy[1]+0.05, r"$\mathrm{\mu}$",fontsize=35)
diagram.text(v2.xy[0], v2.xy[1]-0.05, r"$\mathrm{\nu}$",fontsize=35)
diagram.text(v3.xy[0]-0.07, v3.xy[1]+0.11, r"$q_1$",fontsize=35)
diagram.text(v3.xy[0]-0.07, v3.xy[1]-0.11, r"$q_2$",fontsize=35)

diagram.plot()
fig.savefig('pdf/VBF-LO.pdf',bbox_inches='tight')
