import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ..diagrams import Diagram

@image_comparison(baseline_images=['double'],
                  extensions=['png'], remove_text=True)
def test_double():

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    dia = Diagram(ax)

    v1 = dia.vertex(xy=(.2,.5))
    v2 = dia.vertex(xy=(0.8,.5))
    l1 = dia.line(v1, v2, style="double linear", arrow=True, arrow_param={'linewidth':2, 'width':.035})

    v3 = dia.vertex(xy=(.2,.3))
    v4 = dia.vertex(xy=(0.8,.3))
    l2 = dia.line(v3, v4, style="linear double wiggly")

    v5 = dia.vertex(xy=(.2,.1))
    v6 = dia.vertex(xy=(0.8,.1))
    l3 = dia.line(v5, v6, style="linear double loopy")

    l4 = dia.line(v1, v2, style = "elliptic double wiggly")
    
    dia.plot()
