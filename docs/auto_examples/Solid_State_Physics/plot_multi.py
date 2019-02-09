"""
Electron-phonon self-energy
===========================

Example for multiple independent diagrams.
"""
from feynman import Diagram
import matplotlib.pyplot as plt

def main():

    fig, axes = plt.subplots(figsize=(6,3), nrows=1, ncols=2,
                             subplot_kw=dict(aspect='equal', frameon=True),
                             sharex=True, sharey=True,
                            )

    G_style = dict(style='single', arrow=True, arrow_param={'width':0.05, 'length': .15})
    Ph_style = dict(style='elliptic loopy', ellipse_spread=.55, xamp=0.035, yamp=-0.05, nloops=13)
    DW_style = dict(style='circular loopy', circle_radius=.25, xamp=.04, yamp=.05, nloops=15)
    V_style = dict()

    get_diagram_one(axes[0], G_style, Ph_style, V_style)
    get_diagram_two(axes[1], G_style, DW_style, V_style)

    plt.tight_layout()

    plt.show()

def get_diagram_one(ax, fermion_style, boson_style, vertex_style):

    D = Diagram(ax)

    w = 0.75
    xy0 = [0.5 - w/2, 0.25]
    v1 = D.vertex(xy0, **vertex_style)
    v2 = D.vertex(v1.xy, dx=w, **vertex_style)
    G = D.line(v1, v2, **fermion_style)
    B = D.line(v1, v2, **boson_style)

    # In case the axes get smaller (you have more diagrams), you might want to change the scale
    D.scale(1.0)

    D.plot()
    return D

def get_diagram_two(ax, fermion_style, boson_style, vertex_style):

    D = Diagram(ax)

    w = 0.75
    xy0 = [0.5 - w/2, 0.25]

    v1 = D.vertex(xy0, **vertex_style)
    v2 = D.vertex(v1.xy, dx=w/2, **vertex_style)
    v3 = D.vertex(v1.xy, dx=w, **vertex_style)
    G1 = D.line(v1, v2, **fermion_style)
    G2 = D.line(v2, v3, **fermion_style)
    B = D.line(v2, v2, **boson_style)

    # In case the axes get smaller (you have more diagrams), you might want to change the scale
    D.scale(1.0)

    D.plot()
    return D

if __name__ == '__main__':
    main()
