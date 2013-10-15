
import feynman as fm
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)

ax.set_xlim(0,1)
ax.set_ylim(0,1)
ax.set_xticks([])
ax.set_yticks([])

dia = fm.Diagram(ax)

v1 = dia.verticle(xy=(.2,.5))
v2 = dia.verticle(xy=(0.8,.5))
l1 = dia.line(v1, v2, arrow=True)
l2 = dia.line(v1, v2, pathtype='elliptic', linestyle='loopy')

dia.plot()
plt.show()
