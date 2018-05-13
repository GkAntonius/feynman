
import matplotlib.pyplot as plt
from feynman import Diagram

fig = plt.figure(figsize=(8,2))
ax = fig.add_subplot(111, frameon=False)

ax.set_xlim(0, 2)
ax.set_ylim(0, .5)

W_style = dict(style='double wiggly elliptic', nwiggles=5)
G_style = dict(style='double', arrow=True, arrow_param={'width':0.05})

# Sigma operator
D = Diagram(ax)

v01 = D.vertex([.2, .175])
v02 = D.vertex(v01.xy, dx=.3)

Sigma = D.operator([v01, v02])
Sigma.text("$\Sigma$")

# Equal sign
D.text(v02.x+.2, v02.y, "=", fontsize=30)

# GW convolution
v21 = D.vertex(v02.xy, dxy=[0.4, -0.07])
v22 = D.vertex(v21.xy, dx=0.8)

l21 = D.line(v21, v22, **G_style)
l22 = D.line(v21, v22, **W_style)

l21.text("G", y=.05)
l22.text("W", y=-.1)

D.plot()

# Save
fig.savefig('pdf/gw-Sigma.pdf')
fig.savefig('pdf/gw-Sigma.png')

