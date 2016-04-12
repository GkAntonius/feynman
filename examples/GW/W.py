
import matplotlib.pyplot as plt

from feynman import Diagram

# Set the ax
fig = plt.figure(figsize=(12,3))
ax = fig.add_subplot(111)

ax.set_xlim(0, 3)
ax.set_ylim(0, .75)

y0 = .75 / 2

opwidth = 0.3
linlen = 0.4
tail_marker = 'o'

W_style = dict(style = 'double wiggly', nwiggles=2)
v_style = dict(style = 'simple wiggly', nwiggles=2)

# First diagram
D1 = Diagram(ax)

arrowparam = dict(width=0.05)

xy = [0.2, y0]
v01 = D1.verticle(xy, marker=tail_marker)
xy[0] += linlen
v02 = D1.verticle(v01.xy, dx=linlen, marker=tail_marker)
l01 = D1.line(v01, v02, **W_style)

D1.text(.75, y0, "=", fontsize=30)

xy = [0.9, y0]
v21 = D1.verticle(xy, marker=tail_marker)
v22 = D1.verticle(v21.xy, dx=linlen, marker=tail_marker)
l21 = D1.line(v21, v22, **v_style)

D1.text(1.45, y0, "+", fontsize=30)

xy = [1.6, y0]
v11 = D1.verticle(xy, marker=tail_marker)
v12 = D1.verticle(v11.xy, dx=linlen)
v13 = D1.verticle(v12.xy, dx=opwidth)
v14 = D1.verticle(v13.xy, dx=linlen, marker=tail_marker)

D1.line(v11, v12, **v_style)
D1.line(v13, v14, **W_style)
O = D1.operator([v12,v13], c=1.3)
O.text("P")

D1.plot()

fig.savefig('W.pdf')

