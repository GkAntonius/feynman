import matplotlib

from feynman import Diagram

fig = matplotlib.pyplot.figure(figsize=(1.,1.))
ax = fig.add_axes([0,0,10,10], frameon=False)

diagram = Diagram(ax)
diagram.text(.4,0.9,"Associated Vector Boson", fontsize=40)
diagram.text(.6,0.83,"(VH or 'Higgs Strahlung')", fontsize=40)
in1 = diagram.verticle(xy=(.1,.75), marker='')
in2= diagram.verticle(xy=(.1,.25), marker='')
v1 = diagram.verticle(xy=(.35,.5))
v2 = diagram.verticle(xy=(.65,.5))
higgsout = diagram.verticle(xy=(.9,.75))
out1 = diagram.verticle(xy=(.9,.25),marker='')

q1 = diagram.line(in1, v1)
q2 = diagram.line(v1, in2)
wz1 = diagram.line(v1, v2, style='wiggly')
wz2 = diagram.line(v2, out1, style='wiggly')
higgs = diagram.line(v2, higgsout, arrow=False, style='dashed')

q1.text("q",fontsize=30)
q2.text(r"$\bar{\mathrm{q}}$",fontsize=30)
diagram.text(0.5,0.55,"$Z/W^\pm$",fontsize=30)
diagram.text(0.69,0.35,"$Z/W^\pm$",fontsize=30)
higgs.text("H",fontsize=30)

diagram.plot()
fig.savefig('pdf/VH.pdf',bbox_inches='tight')


