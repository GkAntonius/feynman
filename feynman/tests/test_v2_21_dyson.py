import matplotlib.pyplot as plt
from matplotlib.testing.decorators import image_comparison

from ..diagrams import Diagram

@image_comparison(baseline_images=['simple'],
                  extensions=['png'], remove_text=True)
def test_diagram():

    # Set the ax
    fig = plt.figure(figsize=(10,1.2))
    ax = fig.add_subplot(111)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, .3)
    
    y0 = sum(ax.get_ylim()) / 2
    l = 0.4
    
    x0 = .05
    
    G_style = dict(arrow=True, arrow_param={'width':0.05}, stroke = 'double')
    G0_style = dict(arrow=True, arrow_param={'width':0.05})
    
    # First diagram
    D1 = Diagram(ax)
    
    arrow_param = dict(width=0.05)
    
    x = x0
    v01 = D1.vertex(xy=(x,y0))
    v02 = D1.vertex(v01.xy, dx=l)
    G = D1.line(v01, v02, **G_style)
    
    text_prop = dict(y=0.05, fontsize=20)
    G.text("G", **text_prop)
    
    x = x0 + .55
    D1.text(x, y0, "=", fontsize=30)
    
    x = x0 + .7
    v21 = D1.vertex(xy=(x,y0))
    v22 = D1.vertex(v21.xy, dx=l)
    G0 = D1.line(v21, v22, **G0_style)
    G0.text("G0", **text_prop)
    
    x = x0 + 1.25
    D1.text(x, y0, "+", fontsize=30)
    
    x = x0 + 1.4
    v11 = D1.vertex(xy=(x0 + 1.4,y0))
    v12 = D1.vertex(xy=v11.xy, dx = l)
    v13 = D1.vertex(xy=v12.xy, dx = .2)
    v14 = D1.vertex(xy=v13.xy, dx = l)
    
    D1.line(v11, v12, arrow=True, arrow_param=arrow_param)
    D1.line(v13, v14, stroke='double', arrow=True, arrow_param=arrow_param)
    O = D1.operator([v12,v13])
    O.text("S")

    D1.plot()
