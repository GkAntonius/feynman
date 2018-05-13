"""Diagram class"""

from copy import deepcopy
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

from .. import colors as mc
from .. import vectors
from .. import colors
from .. import core
from ..core import Vertex, Line, Operator

from .plotter import Plotter

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

        self.line_length =  .2
        self.operator_size =  1.5

        self.x0, self.y0 = xy0

        self.vertices = list()
        self.lines = list()
        self.operators = list()

    def vertex(self, xy='auto', **kwargs):
        """
        Create a vertex.

        Arguments
        ---------

        xy :        Coordinates.
        **kwargs :  Any matplotlib line style argument. 

        Returns
        -------
        feynman.core.Vertex instance.
        """
        if xy is 'auto':
            xy = (self.x0, self.y0)
        #else:
        #    if not isinstance(xy, tuple):
        #        raise TypeError()
        #    elif len(xy) != 2:
        #        raise TypeError()
        v = Vertex(xy, **kwargs)
        self.add_vertex(v)
        return v

    def vertices(self, xys, **kwargs):
        """
        Create multiple vertices.

        Arguments
        ---------

        xys :
            List of xy coordinates.

        **kwargs :
            Any matplotlib line style argument. 

        Returns
        -------

        list of feynman.core.Vertex instance.
        """
        xys = np.array(xys)
        if xys.ndim != 2:
             raise ValueError("xy must be a list of xy coordinates.")
 
        vs = list()
        for xy in xys:
            v = Vertex(xy, **kwargs)
            self.add_vertex(v)
            vs.append(v)
        return vs

    def verticle(self, *args, **kwargs):
        warnings.warn('Diagram.verticle is deprecated. ' +
                     'Use Diagram.vertex instead.')
        return self.vertex(*args, **kwargs)

    def verticles(self, *args, **kwargs):
        warning.warn('Diagram.verticles is deprecated. ' +
                     'Use Diagram.vertices instead.')
        return self.vertices(*args, **kwargs)

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

    def add_vertex(self, vertex):
        """Add a feynman.core.Vertex instance."""
        vertex.diagram = self
        self.vertices.append(vertex)

    def add_line(self, line):
        """Add a feynman.core.line instance."""
        line.diagram = self
        self.lines.append(line)

    def add_operator(self, operator):
        """Add an feynman.core.operator instance."""
        operator.diagram = self
        self.operators.append(operator)
        for v in operator.vertices:
            #if v in self.vertices: continue  # Should avoid this check
            self.add_vertex(v)

    def draw(self):
        """Draw the diagram."""

        for v in self.vertices:
            v.draw(self.ax)

        for l in self.lines:
            l.draw(self.ax)

        for O in self.operators:
            O.draw(self.ax)

    def plot(self, *args, **kwargs):
        return self.draw(*args, **kwargs)

    def add_chunk(self, vertex, dx=0, dy=0, angle=0, radius=0, **line_prop):
        """
        Create a chunk going to the vertex by initializing a vertex
        and a line.  The new vertex will be invisible by default.  All other
        keyword arguments are passed to the line.

        Return: new_vertex, new_line
        """
        v_prop = dict(dx=dx, dy=dy, angle=angle, radius=radius)
        v_prop.setdefault('marker', '')
        line_prop.setdefault('style', 'simple single linear')
        line_prop.setdefault('arrow', False)
        v = self.vertex(vertex.xy, **v_prop)
        l = self.line(v, vertex, **line_prop)
        return v, l

    def text(self, *args, **kwargs):
        """Add text using matplotlib.axes.Axes.text."""
        kwargs.setdefault('ha', 'center')
        kwargs.setdefault('va', 'center')
        kwargs.setdefault('fontsize', 30)
        self.ax.text(*args, **kwargs)

    def scale(self, x):
        """Apply a scaling factor."""

        for v in self.vertices:
            v.scale(x)

        for l in self.lines:
            l.scale(x)

        for o in self.operators:
            o.scale(x)


    def get_object_group_limits():
        """
        Return the x0, y0, w, h
        of the leftmost, bottommost, rightmost and topmost objects.
        """
        raise NotImplementedError()


