import matplotlib

from feynman import Diagram

fig = matplotlib.pyplot.figure(figsize=(1.,1.))
ax = fig.add_axes([0,0,10,10], frameon=False)
diagram = Diagram(ax)
#diagram.text(.5,0.9,"Gluon-Gluon Fusion (ggF)",fontsize=40)
in1 = diagram.vertex(xy=(.1,.7), marker='')
in2= diagram.vertex(xy=(.1,.3), marker='')
v1 = diagram.vertex(xy=(.3,.7))
v2 = diagram.vertex(xy=(.3,.3))
v3 = diagram.vertex(xy=(.5,.5))
higgsout = diagram.vertex(xy=(.65,.5))
zout1 = diagram.vertex(xy=(.9,.7), marker='')
zout2 = diagram.vertex(xy=(.9,.3), marker='')

gluon_style = dict(style='linear loopy', xamp=.025, yamp=.035, nloops=4)

g1 = diagram.line(in1, v1, **gluon_style)
g2 = diagram.line(in2, v2, **gluon_style)
t1 = diagram.line(v1, v2)
t2 = diagram.line(v2, v3)
t3 = diagram.line(v3, v1)
higgs = diagram.line(v3, higgsout, arrow=False, style='dashed')
z1 = diagram.line(higgsout, zout1,arrow=False, style='wiggly')
z2 = diagram.line(zout2, higgsout,arrow=False, style='wiggly')

g1.text("$g$",fontsize=30)
g2.text("$g$",fontsize=30)
diagram.text(zout1.xy[0]+.025,zout1.xy[1],"$Z$",fontsize=30)
diagram.text(zout2.xy[0]+.025,zout2.xy[1],"$Z$",fontsize=30)
t1.text("$t$",fontsize=30)
t2.text("$t$",fontsize=30)
t3.text(r"$\bar{t}$",fontsize=30)
higgs.text("H",fontsize=30)
diagram.text(higgsout.xy[0]+0.24,higgsout.xy[1],"$\kappa_\mathrm{SM}/\kappa_{AZZ}/\kappa_{HZZ}$",fontsize=36)

diagram.plot()
fig.savefig('pdf/ggFZZ-EFT.pdf',bbox_inches='tight')
