"""
Dyson equation
==============

A dyson equation.
"""
import matplotlib.pyplot as plt
from feynman import Diagram

# Set up the figure and ax
fig = plt.figure(figsize=(10,1.5))
ax = fig.add_axes([.0,.0,1.,1.], frameon=False)

ax.set_xlim(0, 1.0)
ax.set_ylim(0, .15)

l = 0.15  # Length of the propagator
txt_l = 0.05  # Padding around the symbol
op_l = 0.08  # Size of the operator

G_style = dict(arrow=True, arrow_param={'width':0.02, 'length': 0.05}, style = 'double')
G0_style = dict(arrow=True, arrow_param={'width':0.02, 'length': 0.05}, style = 'simple')
text_prop = dict(y=0.02, fontsize=20)

D = Diagram(ax)

# Left hand side
v11 = D.vertex(xy=[0.05, 0.06])
v12 = D.vertex(v11.xy, dx=l)
G = D.line(v11, v12, **G_style)
G.text("$G$", **text_prop)

# Symbol
D.text(v12.x + txt_l, v12.y, "=")

# First term
v21 = D.vertex(v12.xy, dx=2*txt_l)
v22 = D.vertex(v21.xy, dx=l)
G0 = D.line(v21, v22, **G0_style)
G0.text("$G_0$", **text_prop)

# Symbol
D.text(v22.x + txt_l, v22.y, "+")

# Second term
v31 = D.vertex(v22.xy, dx=2*txt_l)
v32 = D.vertex(v31.xy, dx=l)
v33 = D.vertex(v32.xy, dx=op_l)
v34 = D.vertex(v33.xy, dx=l)
D.line(v31, v32, **G0_style)
D.line(v33, v34, **G_style)
O = D.operator([v32,v33])
O.text("$\Sigma$")

# Plot and show
D.plot()
plt.show()
