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

class TestOperator(TestDiagram):

    def test_ellipse(self):

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # First diagram
        D1 = Diagram(ax)

        v1 = D1.verticle(xy=(.2,.2), marker='')
        v2 = D1.verticle(xy=(.4,.2))
        v3 = D1.verticle(xy=(.6,.2))
        v4 = D1.verticle(xy=(.8,.2), marker='')

        D1.line(v1, v2, arrow=True)
        O = D1.operator([v2,v3])
        O.text("$\Sigma$", -.025,-.025, fontsize=28)
        D1.line(v3, v4, linestyle='double', arrow=True)

        D1.plot()

        # Second diagram
        D2 = Diagram(ax)

        v1 = D2.verticle(xy=(.2,.4), marker='')
        v2 = D2.verticle(xy=(.4,.5))
        v3 = D2.verticle(xy=(.6,.6))
        v4 = D2.verticle(xy=(.8,.7), marker='')

        D2.line(v1, v2, arrow=True)
        D2.operator([v2,v3], c=1.5)
        D2.line(v3, v4, linetype='wiggly', nwiggles=2)

        D2.plot()

        self.show_pdf(basename)
