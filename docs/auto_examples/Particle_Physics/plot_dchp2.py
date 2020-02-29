"""
DCHP2
=====

Doubly Charged Higgs Production
"""
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from feynman import Diagram

fig = matplotlib.pyplot.figure(figsize=(10.,10.))
ax = fig.add_axes([0,0,1,1], frameon=False)

diagram = Diagram(ax)
diagram.text(.4,0.9,"Doubly Charged Higgs Production", fontsize=40)
q1in = diagram.vertex(xy=(.1,.75), marker='')
q2in= diagram.vertex(xy=(.1,.25), marker='')
v1 = diagram.vertex(xy=(.3,.75))
v2 = diagram.vertex(xy=(.3,.25))
vmerge = diagram.vertex(xy=(.6,.5))
higgsout = diagram.vertex(xy=(.8,.5))
q1pout = diagram.vertex(xy=(.95,.75), marker='')
q2pout = diagram.vertex(xy=(.95,.25), marker='')
l1out = diagram.vertex(xy=(.95,.62), marker='')
l2out = diagram.vertex(xy=(.95,.38), marker='')

lw = 5
# Quarks
q1 = diagram.line(q1in, v1, color='blue', lw=lw, arrow_param=dict(color='blue', length=0.08, width=0.02))
q2 = diagram.line(q2in, v2, color='blue', lw=lw, arrow_param=dict(color='blue', length=0.08, width=0.02))
q1p = diagram.line(v1, q1pout, color='blue', lw=lw, arrow_param=dict(color='blue', length=0.08, width=0.02))
q2p = diagram.line(v2, q2pout, color='blue', lw=lw, arrow_param=dict(color='blue', length=0.08, width=0.02))
diagram.text(.05, 0.75, "q", fontsize=40)
diagram.text(.05, 0.25, "q",fontsize=40)
diagram.text(0.98, 0.75, r"$\mathrm{q}^\prime$", fontsize=40)
diagram.text(0.98, 0.25, r"$\mathrm{q}^\prime$",fontsize=40)

# Bosons
w1 = diagram.line(v1, vmerge, style='wiggly', color='green', lw=lw)
w2 = diagram.line(v2, vmerge, style='wiggly', color='green', lw=lw)
higgs = diagram.line(vmerge, higgsout, arrow=False, ls='dashed', lw=lw, dashes=(4, 2))
diagram.text(0.35, 0.6, r"$W^+$", fontsize=40)
diagram.text(0.35, 0.38, r"$W^+$", fontsize=40)
diagram.text(0.72, 0.55, r"$H^{++}$", fontsize=40)

# Leptons
l1 = diagram.line(l1out, higgsout, color='blue', lw=lw, arrow_param=dict(color='blue', length=0.08, width=0.02))
l2 = diagram.line(l2out, higgsout, color='blue', lw=lw, arrow_param=dict(color='blue', length=0.08, width=0.02))
diagram.text(0.98, 0.62, r"$l^+$", fontsize=40)
diagram.text(0.98, 0.38, r"$l^+$", fontsize=40)


diagram.plot()
matplotlib.pyplot.show()
