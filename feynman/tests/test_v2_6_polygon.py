import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ..diagrams import Diagram

@image_comparison(baseline_images=['polygon'],
                  extensions=['png'], remove_text=True)
def test_triangle():

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    dia = Diagram(ax)

    v1 = dia.vertex(xy=(.2,.6), marker='')
    v2 = dia.vertex(xy=(.2,.4), marker='')

    v3 = dia.vertex(xy=(.3,.6))
    v4 = dia.vertex(xy=(.3,.4))

    l1 = dia.line(v1, v3)
    l2 = dia.line(v2, v4)

    v5 = dia.vertex(xy=(.3 + .2 * np.sqrt(3) / 2, .5))

    triangle = dia.operator([v3, v4, v5])

    v6 = dia.vertex(xy=(.8, .5), marker='')
    l3 = dia.line(v5, v6, flavour='wiggly', nwiggles=4)
    
    dia.plot()
