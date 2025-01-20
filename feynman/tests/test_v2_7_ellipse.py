import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ..diagrams import Diagram

@image_comparison(baseline_images=['ellipse'],
                  extensions=['png'], remove_text=True)
def test_ellipse():

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # First diagram
    D1 = Diagram(ax)

    v1 = D1.vertex(xy=(.2,.2), marker='')
    v2 = D1.vertex(xy=(.4,.2))
    v3 = D1.vertex(xy=(.6,.2))
    v4 = D1.vertex(xy=(.8,.2), marker='')

    D1.line(v1, v2, arrow=True)
    O = D1.operator([v2,v3])
    O.text("S")
    D1.line(v3, v4, stroke='double', arrow=True)

    D1.plot()

    # Second diagram
    D2 = Diagram(ax)

    v1 = D2.vertex(xy=(.2,.4), marker='')
    v2 = D2.vertex(xy=(.4,.5))
    v3 = D2.vertex(xy=(.6,.6))
    v4 = D2.vertex(xy=(.8,.7), marker='')

    D2.line(v1, v2, arrow=True)
    D2.operator([v2,v3], c=1.5)
    D2.line(v3, v4, flavour='wiggly', nwiggles=2)

    D2.plot()
