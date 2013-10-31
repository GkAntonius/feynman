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

class TestOperator(TestDiagram):

    def test_ellipse(self):

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        dia = Diagram(ax)

        v1 = dia.verticle(xy=(.2,.5), marker='')
        v2 = dia.verticle(xy=(.4,.5))
        v3 = dia.verticle(xy=(.6,.5))
        v4 = dia.verticle(xy=(.8,.5), marker='')


        l1 = dia.line(v1, v2, arrow=True)
        l2 = dia.line(v3, v4, linestyle='wiggly', nwiggles=4)

        o1 = dia.operator([v2,v3])

        dia.plot()

        self.show_pdf(basename)
