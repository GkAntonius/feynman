import numpy as np
import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ..diagrams import Diagram

@image_comparison(baseline_images=['sigma'],
                  extensions=['png'], remove_text=True)
def test_diagram():

    # Set the ax
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 1.5)
    
    x0 = .2
    y0 = sum(ax.get_ylim()) / 2
    
    opwidth = 0.3
    linlen = 0.8
    tail_marker = 'o'
    Gamma_width = .3
    
    W_style = dict(style='double wiggly ellipse', nwiggles=4)
    G_style = dict(style='double', arrow=True, arrow_param={'t':.5, 'width':0.05})
    
    # First diagram
    D1 = Diagram(ax)
    
    v1 = D1.vertex([x0, y0])
    v2 = D1.vertex(v1.xy, dx=opwidth)
    Sigma = D1.operator([v1,v2])
    Sigma.text("S")
    
    x0 = v2.xy[0] + .2
    D1.text(x0, y0, "=", fontsize=30)
    
    x0 += .2
    y0 -= .1
    
    v1 = D1.vertex((x0, y0))
    v2 = D1.vertex(v1.xy, dx= (linlen - Gamma_width / 2))
    v4 = D1.vertex(v2.xy, dx=Gamma_width)
    v3 = D1.vertex(v4.xy, dxy=(- Gamma_width / 2, Gamma_width * np.sqrt(3) / 2))
    l1 = D1.line(v1, v2, **G_style)
    l2 = D1.line(v1, v3, **W_style)
    
    Gamma = D1.operator([v2,v3, v4])
    Gamma.text("G")
    
    D1.plot()
