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
        fig = plt.figure(figsize=(10,4))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0, 2.5)
        ax.set_ylim(0, 1.)

        # First diagram
        D1 = Diagram(ax)

        v01 = D1.verticle(xy=(.2,.5), marker='')
        v02 = D1.verticle(xy=(.6,.5), marker='')
        l01 = D1.line(v01, v02, arrow=True, linestyle = 'double')

        v01.text('1')
        v02.text('2')

        D1.text(.75, .5, "=", fontsize=30)

        x0 = .7
        v21 = D1.verticle(xy=(x0+.2,.5), marker='')
        v22 = D1.verticle(xy=(x0+.6,.5), marker='')
        l21 = D1.line(v21, v22, arrow=True, linestyle = 'simple')

        v21.text('1')
        v22.text('2')

        D1.text(1.45, .5, "+", fontsize=30)

        x0 = 1.4
        v11 = D1.verticle(xy=(x0+.2,.5), marker='')
        v12 = D1.verticle(xy=(x0+.4,.5))
        v13 = D1.verticle(xy=(x0+.6,.5))
        v14 = D1.verticle(xy=(x0+.8,.5), marker='')

        v11.text('1')
        v12.text('3', x=-.04, y=.025)
        v13.text('4', x= .01, y=.025)
        v14.text('2')


        D1.line(v11, v12, arrow=True)
        D1.line(v13, v14, linestyle='double', arrow=True)
        O = D1.operator([v12,v13])
        O.text("$\Sigma$")

        D1.plot()

        self.show_pdf(basename)



