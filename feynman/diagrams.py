#
# TODO
# ----
#
#   o Verticle
#       - get_lines
#       - get_operators
#
#   o Line, Operator
#       - get_verticles
#
#   o Diagram
#       - add_text
#       - verticles (add multiple verticles at once)
#
#
#   o Scalability
#
# =========================================================================== #

from copy import deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

# Personnal modules
import mycolors as mc

from . import vectors
from . import colors
from . import core

from core import Verticle, Line, Operator

class Diagram(object):
    """
    The main object for a feynman diagram.

    Arguments
    ---------

    fig : A :class:`matplotlib.figure.Figure`  instante.
    ax : A :class:`matplotlib.axes.AxesSubplot`  instance.
"""

    _scale = (1., 1.)
    _transform = None

    def __init__(self, ax=None, xy0=(0.,0.), set_ticks=True):

        if ax is not None:
            self.ax = ax
            if set_ticks:
                self.ax.set_xticks([])
                self.ax.set_yticks([])
        else:
            fig = plt.figure(figsize=(6,6))
            self.ax = fig.gca()
            self.ax.set_xlim(0,1)
            self.ax.set_ylim(0,1)

        self.verticles = list()
        self.lines = list()
        self.operators = list()

    def verticle(self, xy=(0,0), **kwargs):
        """
        Create a verticle.

        Arguments
        ---------

        xy :        Coordinates.
        **kwargs :  Any matplotlib line style argument. 

        Returns
        -------
        feynman.core.Verticle instance.
"""
        v = Verticle(xy, **kwargs)
        self.verticles.append(v)
        return v

    def verticles(self, xys, **kwargs):
        """
        Create multiple verticles.

        Arguments
        ---------

        xys :
            List of xy coordinates.

        **kwargs :
            Any matplotlib line style argument. 

        Returns
        -------

        list of feynman.core.Verticle instance.
"""
        xys = np.array(xys)
        if xys.ndim != 2:
             raise ValueError("xy must be a list of xy coordinates.")
 
        vs = list()
        for xy in xys:
            v = Verticle(xy, **kwargs)
            self.verticles.append(v)
            vs.append(v)
        return vs

    def line(self, *args, **kwargs):
        """Create a feynman.core.line instance."""
        l = Line(*args, **kwargs)
        self.lines.append(l)
        return l

    def operator(self, *args, **kwargs):
        """Create an feynman.core.operator instance."""
        O = Operator(*args, **kwargs)
        self.operators.append(O)
        return O

    def add_verticle(self, verticle):
        """Add a feynman.core.verticle instance."""
        self.verticles.append(verticle)

    def add_line(self, line):
        """Add a feynman.core.line instance."""
        self.lines.append(line)

    def add_operator(self, operator):
        """Add an feynman.core.operator instance."""
        self.operators.append(operator)

    def plot(self):
        """Draw the diagram."""

        for v in self.verticles:
            v.draw(self.ax)

        for l in self.lines:
            l.draw(self.ax)

        for O in self.operators:
            O.draw(self.ax)

    def text(self, *args, **kwargs):
        """Add text using matplotlib.axes.Axes.text."""
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'center')
        kwargs.setdefault('fontsize', 30)
        self.ax.text(*args, **kwargs)

    def show(self):
        """Show the figure with matplotlib.pyplot.show."""
        plt.show()

    def gcf(self):
        """Get the figure."""
        return self.fig

    def gca(self):
        """Get the axe."""
        return self.ax
