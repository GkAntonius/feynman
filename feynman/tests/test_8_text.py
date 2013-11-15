"""
Create two lines. One straight and one elliptic.
"""

import unittest
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from ..core import Diagram

from . import TestDiagram

basename = os.path.splitext(os.path.basename(__file__))[0]

class TestText(TestDiagram):

    def test_text(self):

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

        v1.text('1', x=-.03, y=.02)
        v2.text('2', x=-.03, y=.02)
        v3.text('3', x= .01, y=.02)
        v4.text('4', x= .01, y=.02)

        D1.line(v1, v2, arrow=True)
        O = D1.operator([v2,v3])
        O.text("$\Sigma$", -.025,-.025, fontsize=28)
        D1.line(v3, v4, linestyle='double', arrow=True)

        D1.plot()

        # Second diagram
        D2 = Diagram(ax)

        v1 = D2.verticle(xy=(.1,.5), marker='')
        v2 = D2.verticle(xy=(.3,.5))
        v3 = D2.verticle(xy=(.7,.5))
        v4 = D2.verticle(xy=(.9,.5), marker='')

        l12 = D2.line(v1, v2, arrow=True)
        w23 = D2.line(v2, v3, pathtype='elliptic', linestyle='loopy',
                nloops=18)
        l23 = D2.line(v2, v3, arrow=True)
        l34 = D2.line(v3, v4, arrow=True)

        l12.text("$\\vec{p}$",          t=.5,  y=-.05, fontsize=18)
        w23.text("$\\vec{q}$",          t=.48, y=-.05, fontsize=18)
        l23.text("$\\vec{p}-\\vec{q}$", t=.35,  y=-.06, fontsize=18)
        l34.text("$\\vec{p}$",          t=.5,  y=-.05, fontsize=18)
        
        D2.plot()

        self.show_pdf(basename)
