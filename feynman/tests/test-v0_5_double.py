"""
Create two lines. One straight and one elliptic.
"""

import unittest
import os
import time
import matplotlib.pyplot as plt
from ..diagrams import Diagram

from . import TestDiagram

basename = os.path.splitext(os.path.basename(__file__))[0]

class TestLines(TestDiagram):

    def test_double(self):

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        dia = Diagram(ax)

        v1 = dia.verticle(xy=(.2,.5))
        v2 = dia.verticle(xy=(0.8,.5))
        l1 = dia.line(v1, v2, pathtype='linear', linestyle='double',
                      arrow=True, arrowparam={'linewidth':2, 'size':.035})

        v3 = dia.verticle(xy=(.2,.3))
        v4 = dia.verticle(xy=(0.8,.3))
        l2 = dia.line(v3, v4,
                    pathtype='linear', linestyle='double', linetype='wiggly',)

        v5 = dia.verticle(xy=(.2,.1))
        v6 = dia.verticle(xy=(0.8,.1))
        l3 = dia.line(v5, v6,
                    pathtype='linear', linestyle='double', linetype='loopy',)

        l4 = dia.line(v1, v2,
                    pathtype='elliptic', linestyle='double', linetype='wiggly',)

        
        dia.plot()

        self.show_pdf(basename)
