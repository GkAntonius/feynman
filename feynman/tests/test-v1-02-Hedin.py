"""

Create two lines. One straight and one elliptic.




"""
import unittest
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from ..diagrams import Diagram

from . import TestDiagram

basename = os.path.splitext(os.path.basename(__file__))[0]

class TestText(TestDiagram):

    def test_diagram(self):

        # Set the ax
        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0,3)
        ax.set_ylim(0,2)

        # TODO should be set by diagram.
        #ax.set_xticks([])
        #ax.set_yticks([])
        
        # First diagram
        D1 = Diagram(ax)

        y0 = 1.7

        v01 = D1.verticle(xy=(.2,y0), marker='')
        v02 = D1.verticle(xy=(.6,y0), marker='')
        l01 = D1.line(v01, v02, arrow=True, linestyle = 'double')

        D1.text(.7, y0-.03, "=", fontsize=36)
        D1.text(1.4, y0-.03, "+", fontsize=36)

        x0 = .7
        v21 = D1.verticle(xy=(x0+.2,y0), marker='')
        v22 = D1.verticle(xy=(x0+.6,y0), marker='')
        l21 = D1.line(v21, v22, arrow=True, linestyle = 'simple')

        x0 = 1.4
        v11 = D1.verticle(xy=(x0+.2,y0), marker='')
        v12 = D1.verticle(xy=(x0+.4,y0))
        v13 = D1.verticle(xy=(x0+.6,y0))
        v14 = D1.verticle(xy=(x0+.8,y0), marker='')


        D1.line(v11, v12, arrow=True)
        D1.line(v13, v14, linestyle='double', arrow=True)
        O = D1.operator([v12,v13])
        O.text("$\Sigma$", -.025, -.025, fontsize=28)

        D1.plot()

        self.show_pdf(basename)



