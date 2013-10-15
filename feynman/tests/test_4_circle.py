"""
Create two lines. One straight and one elliptic.
"""

import unittest
import os
import time
import matplotlib.pyplot as plt
from ..core import Diagram

from . import TestDiagram

class TestLines(TestDiagram):

    def test_simple(self):

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        dia = Diagram(ax)
        
        v1 = dia.verticle(xy=(.2,.2), marker='')
        v2 = dia.verticle(xy=(0.5,.2))
        v3 = dia.verticle(xy=(0.8,.2), marker='')
        l1 = dia.line(v1, v2, arrow=True)
        l2 = dia.line(v2, v2, pathtype='circular', linestyle='simple')
        l3 = dia.line(v2, v3, arrow=True)
        
        dia.plot()

        self.show_pdf()

    def test_wiggly(self):

        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111)
        
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        dia = Diagram(ax)
        
        v1 = dia.verticle(xy=(.2,.2), marker='')
        v2 = dia.verticle(xy=(0.5,.2))
        v3 = dia.verticle(xy=(0.8,.2), marker='')
        l1 = dia.line(v1, v2, arrow=True)
        l2 = dia.line(v2, v2, pathtype='circular', linestyle='wiggly',
                      nwiggles=7, phase=.25)
        l3 = dia.line(v2, v3, arrow=True)
        
        dia.plot()

        self.show_pdf()

