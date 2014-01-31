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
#       - text as an option on init
#       - verticles (add multiple verticles at once)
#       - set_center
#       - set_angles
#       - get_dimensions
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

from .. import colors as mc
from .. import vectors
from .. import colors
from .. import core
from ..core import Verticle, Line, Operator

from .plotter import Plotter

# Personnal modules

class Diagram(Plotter):
    """
    The main object for a feynman diagram.

    Arguments
    ---------

    fig : A :class:`matplotlib.figure.Figure`  instante.
    ax : A :class:`matplotlib.axes.AxesSubplot`  instance.

    transparent : True to set the background as transparent. 


"""

    _scale = (1., 1.)
    _transform = None

    def __init__(self, ax=None, xy0=(0.,0.), **kwargs):

        self._init_figure(ax=ax, **kwargs)

        self._init_objects()

        self.x0 = .2
        self.y0 = .2
        self.line_length =  .2
        self.operator_size =  1.5

    def _init_objects(self):
        """Init lists of objects."""
        self.verticles = list()
        self.lines = list()
        self.operators = list()


    def set_size_inches(self, w=8, h=6):
        """Set the figure size, and set xlim, ylim, x0 and y0 accordingly."""
        # Geometry
        w, h = 8, 6
        aspectratio = float(h) / float(w)
        self.fig.set_size_inches(w, h)

        self.ax.set_xlim(.0, w)

        #self.ax.set_xlim(.0, 10.)
        self.ax.set_ylim(np.array(self.ax.get_xlim()) * aspectratio)
        self.y0 = sum(self.ax.get_ylim()) / 2.
        self.x0 = sum(self.ax.get_xlim()) * .05


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
        for v in self.verticles: v.draw(self.ax)
        for l in self.lines: l.draw(self.ax)
        for O in self.operators: O.draw(self.ax)

    def text(self, *args, **kwargs):
        """Add text using matplotlib.axes.Axes.text."""
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'center')
        kwargs.setdefault('fontsize', 30)
        self.ax.text(*args, **kwargs)

    # ======================================================================= #

    # Geometry ============================================================== #


    # FIXME
    def get_object_group_limits():
        """
        Return the x0, y0, w, h
        of the leftmost, bottommost, rightmost and topmost objects.
        """
        raise NotImplementedError()

    # ======================================================================= #
