
import os

import numpy as np
import matplotlib.pyplot as plt

from feynman import Diagram
from myfunctions import openfile

# Set the ax
fig = plt.figure(figsize=(10,1.2))
ax = fig.add_subplot(111, frameon=False)

ax.set_xlim(0, 2.5)
ax.set_ylim(0, .3)

y0 = sum(ax.get_ylim()) / 2
l = 0.4

x0 = .05

G_style = dict(arrow=True, arrow_param={'width':0.05}, style = 'double')
G0_style = dict(arrow=True, arrow_param={'width':0.05}, style = 'simple')

D = Diagram(ax)

x = x0
v01 = D.verticle(xy=(x,y0))
v02 = D.verticle(v01.xy, dx=l)
G = D.line(v01, v02, **G_style)

text_prop = dict(y=0.05, fontsize=20)
G.text("$G$", **text_prop)

x = x0 + .55
D.text(x, y0, "=", fontsize=30)

x = x0 + .7
v21 = D.verticle(xy=(x,y0))
v22 = D.verticle(v21.xy, dx=l)
G0 = D.line(v21, v22, **G0_style)
G0.text("$G_0$", **text_prop)

x = x0 + 1.25
D.text(x, y0, "+", fontsize=30)

x = x0 + 1.4
v11 = D.verticle(xy=(x,y0))
v12 = D.verticle(xy=(x+l,y0))
v13 = D.verticle(xy=(x+l+.2,y0))
v14 = D.verticle(xy=(x+l+.2+l,y0))


D.line(v11, v12, **G0_style)
D.line(v13, v14, **G_style)
O = D.operator([v12,v13])
O.text("$\Sigma$")

D.plot()

fig.savefig('pdf/hedin-dyson.pdf')

