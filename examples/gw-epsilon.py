
import numpy as np
import matplotlib.pyplot as plt

from feynman import Diagram

# Set the ax
fig = plt.figure(figsize=(12,3))
ax = fig.add_subplot(111, frameon=False)

ax.set_xlim(0, 3)
ax.set_ylim(0, .75)

y0 = .75 / 2

side = 0.3
gammalen = side * np.sqrt(3) / 2
linlen = 0.4
opwidth = 0.3
obj_spacing = .23
tail_marker = 'o'

W_style = dict(style='double wiggly', nwiggles=2)
v_style = dict(style='simple wiggly', nwiggles=2)
G_style = dict(style='double',  arrow=True, arrow_param={'width':0.05})

D = Diagram(ax)

xy = [0.2, y0]

v01 = D.verticle(xy)
v02 = D.verticle(v01.xy, dx=opwidth)
epsilon = D.operator([v01,v02], c=1.1)
epsilon.text("$\\varepsilon$", fontsize=50)

D.text(v02.xy[0]+obj_spacing, y0, "=", fontsize=30)

v30 = D.verticle(v02.xy, dx=2*obj_spacing)

n1 = np.array([-1.,0.])
n2 = np.array([ 1.,0.])

chunkdist = .03
v310 = D.verticle(v30.xy, dxy=n1*chunkdist, marker='')
v320 = D.verticle(v30.xy, dxy=n2*chunkdist, marker='')

chunklen = .025
v31 = D.verticle(v310.xy, dxy=n1*chunklen, marker='')
v32 = D.verticle(v320.xy, dxy=n2*chunklen, marker='')

chunkstyle=dict(arrow=False, linewidth=6.)
D.line(v310, v31, **chunkstyle)
D.line(v320, v32, **chunkstyle)


D.text(v30.xy[0] + obj_spacing, y0, "-", fontsize=40)


xy = [1.6, y0]
v11 = D.verticle(v30.xy, dx=2 * obj_spacing)
v12 = D.verticle(v11.xy, dx=linlen)
v13 = D.verticle(v12.xy, dx=opwidth)

D.line(v11, v12, **v_style)
O = D.operator([v12,v13], c=1.3)

O.text("$P$")

D.plot()

fig.savefig('pdf/gw-epsilon.pdf')

