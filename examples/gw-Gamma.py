
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
tail_marker = 'o'

W_style = dict(style = 'double wiggly', nwiggles=2)
v_style = dict(style = 'simple wiggly', nwiggles=2)
G_style = dict(style = 'double',  arrow=True, arrow_param={'width':0.05})

D = Diagram(ax)

arrow_param = dict(width=0.05)

xy = [0.2, y0]

v01 = D.verticle(xy, dy= side/2)
v02 = D.verticle(xy, dy=-side/2)
v03 = D.verticle(xy, dx=gammalen)
gamma0 = D.operator([v01,v02,v03])
gamma0.text("$\Gamma$")

D.text(.75, y0, "=", fontsize=30)

v30 = D.verticle([1.05, y0])


n1 = np.array([-np.sqrt(3)/6, .5])
n2 = np.array([-np.sqrt(3)/6,-.5])
n3 = np.array([ np.sqrt(3)/3, .0])

chunkdist = .05
v310 = D.verticle(v30.xy, dxy=n1*chunkdist, marker='')
v320 = D.verticle(v30.xy, dxy=n2*chunkdist, marker='')
v330 = D.verticle(v30.xy, dxy=n3*chunkdist, marker='')

chunklen = .05
v31 = D.verticle(v310.xy, dxy=n1*chunklen, marker='')
v32 = D.verticle(v320.xy, dxy=n2*chunklen, marker='')
v33 = D.verticle(v330.xy, dxy=n3*chunklen, marker='')

chunkstyle=dict(arrow=False, linewidth=6.)
D.line(v310, v31, **chunkstyle)
D.line(v320, v32, **chunkstyle)
D.line(v330, v33, **chunkstyle)


D.plot()

fig.savefig('pdf/gw-Gamma.pdf')

