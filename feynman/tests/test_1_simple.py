"""
Create two lines. One straight and one elliptic.
"""

import unittest
import os
import time
import matplotlib.pyplot as plt
from ..core import Diagram

from . import TestDiagram

basename = os.path.splitext(os.path.basename(__file__))[0]

class TestLines(TestDiagram):

    def test_simple(self):

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        dia = Diagram(ax)
        
        v11 = dia.verticle(xy=(.2,.1))
        v21 = dia.verticle(xy=(0.8,.1))
        l11 = dia.line(v11, v21)

        v1 = dia.verticle(xy=(.2,.5))
        v2 = dia.verticle(xy=(0.8,.5))
        l1 = dia.line(v1, v2, pathtype='elliptic', ellipse_excentricity=1.5)
        l2 = dia.line(v2, v1, pathtype='elliptic', ellipse_excentricity=1.5)
        #l2 = dia.line(v2, v1)
        print ''
        for l in (l1, l2):
            print l.rstart, l.rend
        print 'linepath:', l1.linepath[0], l2.linepath[-1]
        print 'tangent:', l1.tangent[0], l2.tangent[-1]

        dia.plot()

        self.show_pdf(basename)

