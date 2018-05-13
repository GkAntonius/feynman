
import matplotlib.pyplot as plt

from feynman import Diagram

# Set the ax
fig = plt.figure(figsize=(12,3))
ax = fig.add_subplot(111, frameon=False)

ax.set_xlim(0, 3)
ax.set_ylim(0, .75)

y0 = .75 / 2

opwidth = 0.3
linlen = 0.4
tail_marker = 'o'

W_style = dict(style='double wiggly', nwiggles=2)
v_style = dict(style='simple wiggly', nwiggles=2)

D = Diagram(ax)

arrowparam = dict(width=0.05)

xy = [0.2, y0]
v01 = D.vertex(xy, marker=tail_marker)
xy[0] += linlen
v02 = D.vertex(v01.xy, dx=linlen, marker=tail_marker)
W = D.line(v01, v02, **W_style)

text_prop = dict(y=0.06, fontsize=22)

W.text("$W$", **text_prop)

D.text(.75, y0, "=", fontsize=30)

xy = [0.9, y0]
v21 = D.vertex(xy, marker=tail_marker)
v22 = D.vertex(v21.xy, dx=linlen, marker=tail_marker)
v = D.line(v21, v22, **v_style)
v.text("$v$", **text_prop)

D.text(1.45, y0, "+", fontsize=30)

xy = [1.6, y0]
v11 = D.vertex(xy, marker=tail_marker)
v12 = D.vertex(v11.xy, dx=linlen)
v13 = D.vertex(v12.xy, dx=opwidth)
v14 = D.vertex(v13.xy, dx=linlen, marker=tail_marker)

D.line(v11, v12, **v_style)
D.line(v13, v14, **W_style)
O = D.operator([v12,v13], c=1.3)
O.text("$P$")

D.plot()

fig.savefig('pdf/hedin-W.pdf')
