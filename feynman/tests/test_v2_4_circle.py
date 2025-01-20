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
    
    v1 = D.vertex(xy=(.2,.2), marker='')
    v2 = D.vertex(xy=(0.5,.2))
    v3 = D.vertex(xy=(0.8,.2), marker='')
    v4 = D.vertex(xy=(0.5,.6))
    l1 = D.line(v1, v2, arrow=True)
    l3 = D.line(v2, v4, flavour='wiggly', nwiggles=3)
    l3 = D.line(v2, v3, arrow=True)
    l2 = D.line(v4, v4, shape='circular', flavour='simple')
    
    D.plot()

@image_comparison(baseline_images=['wiggly'],
                  extensions=['png'], remove_text=True)
def test_wiggly():

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    D = Diagram(ax)
    
    v1 = D.vertex(xy=(.2,.2), marker='')
    v2 = D.vertex(xy=(0.5,.2))
    v3 = D.vertex(xy=(0.8,.2), marker='')
    l1 = D.line(v1, v2, arrow=True)
    l2 = D.line(v2, v2, shape='circular', flavour='wiggly',)
    l3 = D.line(v2, v3, arrow=True)
    
    D.plot()

@image_comparison(baseline_images=['angle'],
                  extensions=['png'], remove_text=True)
def test_angle():

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xticks([])
    ax.set_yticks([])
    
    D = Diagram(ax)
    
    v1 = D.vertex((.2,.2))
    v2 = D.vertex((0.5,.7))
    l1 = D.line(v1, v2, flavour='wiggly')
    D.line(v2, v2, shape='circular', arrow=True, circle_angle=l1.angle)
    
    D.plot()

