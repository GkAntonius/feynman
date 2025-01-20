import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ..diagrams import Diagram

@image_comparison(baseline_images=['loopy'],
                  extensions=['png'], remove_text=True)
def test_loopy():

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    D = Diagram(ax)
    
    v1 = D.vertex(xy=(.2,.5))
    v2 = D.vertex(xy=(0.8,.5))
    l1 = D.line(v1, v2, arrow=True)
    l2 = D.line(v1, v2, shape='elliptic', flavour='loopy', nloops=22)
    v3 = D.vertex(xy=(.2,.2))
    v4 = D.vertex(xy=(0.8,.2))
    l3 = D.line(v3, v4, shape='linear', flavour='loopy')

    D.plot()

