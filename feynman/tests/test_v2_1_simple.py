import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ..diagrams import Diagram

@image_comparison(baseline_images=['simple'],
                  extensions=['png'], remove_text=True)
def test_simple():

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    D = Diagram(ax)

    v1 = D.vertex(xy=(.2,.1))
    v2 = D.vertex(xy=(0.8,.1))
    l1 = D.line(v1, v2)
    v1 = D.vertex(xy=(.2,.5))
    v2 = D.vertex(xy=(0.8,.5))
    l1 = D.line(v1, v2, shape='elliptic', ellipse_excentricity=1.5)
    l2 = D.line(v2, v1, shape='elliptic', ellipse_excentricity=1.5)

    D.plot()
