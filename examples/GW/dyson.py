
import matplotlib.pyplot as plt

from feynman import Diagram

# Set the ax
fig = plt.figure(figsize=(10,1.2))
ax = fig.add_subplot(111)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_xlim(0, 2.5)
ax.set_ylim(0, .3)

y0 = sum(ax.get_ylim()) / 2
l = 0.4

x0 = .05

tail_marker = 'o'

# First diagram
D1 = Diagram(ax)

arrow_param = dict(width=0.05)

x = x0
v01 = D1.verticle(xy=(x,y0), marker=tail_marker)
v02 = D1.verticle(xy=(x + l,y0), marker=tail_marker)
l01 = D1.line(v01, v02, arrow=True, arrow_param=arrow_param, style = 'double')

x = x0 + .55
D1.text(x, y0, "=", fontsize=30)

x = x0 + .7
v21 = D1.verticle(xy=(x,y0), marker=tail_marker)
v22 = D1.verticle(xy=(x+l,y0), marker=tail_marker)
l21 = D1.line(v21, v22, arrow=True, arrow_param=arrow_param, style = 'simple')

x = x0 + 1.25
D1.text(x, y0, "+", fontsize=30)

x = x0 + 1.4
v11 = D1.verticle(xy=(x,y0), marker=tail_marker)
v12 = D1.verticle(xy=(x+l,y0))
v13 = D1.verticle(xy=(x+l+.2,y0))
v14 = D1.verticle(xy=(x+l+.2+l,y0), marker=tail_marker)


D1.line(v11, v12, arrow=True, arrow_param=arrow_param)
D1.line(v13, v14, style='double', arrow=True, arrow_param=arrow_param)
O = D1.operator([v12,v13])
O.text("$\Sigma$")

D1.plot()
fig.savefig('Dyson.pdf')

