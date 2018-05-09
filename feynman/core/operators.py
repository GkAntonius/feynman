
from copy import deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

from . import Drawable
from . import Line, Verticle

from .. import vectors
from .. import colors
from ..constants import tau

__all__ = ['Operator', 'RegularOperator', 'RegularBubbleOperator']

class Operator(Drawable):
    """
    A N-point operator.
    Often represented as a polygon, or a circle.


    Arguments
    ---------

    verticles : a list of N verticles (feynman.Verticle)
        First and second verticle, counted clockwise
        defining an edge (or the starting and ending points)
        of a patch object. 

    rotate : (0.)
        Rotation angle to the operator, in units of tau.

    c : (1.)
        If the shape is elliptic, c is the excentricity of the ellipse,
        that is, the ratio of the long axe over the short axe.
        When c = 1, the shape will be a circle.

    shape :
        By default, the shape is 'ellipse' if N=2,
        and 'polygon' if N>2.
        When N=4 however, you can also specify 'bubble' as a shape
        to have lines connecting (v1, v2) together and (v3, v4) together,
        and a single elliptic patch on top of those two lines.

    line_prop :
        Line properties if shape='bubble'.

    **kwargs :
        Any other style specification for a matplotlib.patches.Patch instance.

    Properties
    ----------

    verticles :
        list of N feynman.Verticle

    N :
        Number of verticles to the operator.

    """
    def __init__(self, verticles, **kwargs):

        # Default values
        default = dict(
            c=1,
            )

        # Set default values
        for key, val in default.items():
            kwargs.setdefault(key, val)

        self.verticles = verticles
        self.N = len(verticles)

        # TODO enforce possible values
        shape = kwargs.pop('shape', 'polygon')
        if self.N == 2:
            self.shape = 'ellipse'
        elif self.N == 4:
            self.shape = shape
        else:
            self.shape = 'polygon'
            #self.shape = kwargs.setdefault('shape', 'circle')

        self.ellipse_excentricity = kwargs.pop('c')

        self.line_prop = kwargs.pop('line_prop', dict())

        self.style = dict(
            edgecolor="k",
            facecolor=colors.lightgrey,
            linewidth=3,
            )

        self.style.update(kwargs)

        self.texts = list()
        self.lines = list()

    def get_verticles(self):
        """Return the verticles."""
        return self.verticles

    def set_verticles(self, *verticles):
        """Return the verticles."""
        self.verticles = verticles

    def set_angles(self, *angles):
        """Set the angles between verticles."""
        raise NotImplementedError()

    def set_center(self, xy):
        """Set the center of the polygon """
        raise NotImplementedError()

    @classmethod
    def _check_verticle_distances(verticles, tolerance=1e-8):
        """Assert that all verticles are equally distant from the center."""

    def get_xy(self):
        """Return the xy coordinates of the verticles, clockwise."""
        return np.array([v.xy for v in self.verticles])

    # TODO make it a property
    def get_center(self):
        """Return the xy coordinates of the center."""
        center = np.array([0., 0.])
        for xy in self.get_xy():
            center += xy
        center /= self.N
        return center

    def get_patch(self, *args, **kwargs):
        """Return the patch object"""
        if self.shape.lower() == "ellipse":
            return self.get_ellipse(*args, **kwargs)
        elif self.shape.lower() == "polygon":
            return self.get_polygon(*args, **kwargs)
        elif self.shape.lower() == "bubble":
            return self.get_bubble(*args, **kwargs)
        else:
            raise ValueError("Unrecognized shape: " + self.shape)

    def get_polygon(self):
        """Return the polygon."""
        polygon = mpa.Polygon(self.get_xy(), **self.style)
        return polygon

    def get_ellipse(self):
        """Return an oval between two vertices."""
        start, end = self.get_xy()
        dxy = end - start
        width = np.linalg.norm(dxy)
        height = width / self.ellipse_excentricity
        center = self.get_center()
        angle = vectors.angle(dxy, 'deg')
        ellipse = mpa.Ellipse(center, width, height, angle=angle, **self.style)
        return ellipse

    def get_bubble(self):
        """Return an oval on top of the lines between [v1."""
        xys = self.get_xy()
        vwidth  = abs(xys[1][0] - xys[0][0])
        vheight = abs(xys[0][1] - xys[-1][1])
        width = 0.6 * vwidth
        height = 1.1 * vheight
        center = self.get_center()
        angle = vectors.angle(xys[1] - xys[0], 'deg')
        ellipse = mpa.Ellipse(center, width, height, angle=angle, zorder=2, **self.style)
        return ellipse

    def get_lines(self):
        """Return the lines used for the bubble shape."""
        if not self.shape.lower() == "bubble":
            return list()

        default_style = dict(
            style = 'simple straight linear',
            arrow = False,
            zorder = 1,
            )

        for key, val in default_style.items():
            self.line_prop.setdefault(key, val)

        v1, v2, v3 ,v4 = self.verticles
        line1 = Line(v1, v2, **self.line_prop)
        line2 = Line(v3, v4, **self.line_prop)

        return line1, line2

    def text(self, s, x=0., y=0., **kwargs):
        """
        Add text in the operator.

        Arguments
        ---------

        s : Text string.

        Positional Arguments
        -----------------

        x : (0.)
            x position, relative to the center of the operator.

        y : (0.)
            y position, relative to the center of the operator.

        Keyword Arguments
        -----------------

        fontsize : (30)
            The font size.

        **kwargs :
            Any other style specification for a matplotlib.text.Text instance.
        """
        default = dict(fontsize=30, zorder=10)
        for key, val in default.items():
            kwargs.setdefault(key, val)
        self.texts.append((s, x, y, kwargs))

    def get_texts(self):
        """Return a list of matplotlib.text.Text instances."""
        texts = list()
        for (s, x, y, kwargs) in self.texts:
            kwargs.setdefault('ha', 'center')
            kwargs.setdefault('va', 'center')
            center = self.get_center()
            xtext, ytext = center + np.array([x,y])
            texts.append(mpt.Text(xtext, ytext, s, **kwargs))
        return texts

    def draw(self, ax):
        """Draw the diagram."""
        for line in self.get_lines():
            line.draw(ax)
        patch = self.get_patch()
        ax.add_patch(patch)
        for text in self.get_texts():
            ax.add_artist(text)
        return


class RegularOperator(Operator):
    """
    A N-point operator.
    Often represented as a polygon, or a circle.


    Arguments
    ---------

    N: Number of verticles

    angle:
        Verticles, are counted clockwise. angle=0=1 means
        that there is a verticle in the [1,0] direction from the center.

    rotate : (0.)
        Rotation angle to the operator, in units of tau.

    c : (1.)
        If the shape is elliptic, c is the excentricity of the ellipse,
        that is, the ratio of the long axe over the short axe.
        When c = 1, the shape will be a circle.

    **kwargs :
        Any other style specification for a matplotlib.patches.Patch instance.

    Returns
    -------

    Vs : list of N verticle.


    Properties
    ----------

    verticles :
        list of N feynman.Verticle

    N :
        Number of verticles to the operator.

    """
    def __init__(self, N, center, size=0.3, angle=None, **kwargs):

        # TODO: add keyword variable side to specify the distance between vertices rather than size.

        if angle is None:
            if N > 2:
                angle = tau / (2 * N)
            else:
                angle = 0.

        vertices = list()
        for i in range(N):
            v = Verticle(xy=center, radius=size, angle=angle + float(i)/N)
            vertices.append(v)

        kwargs.update(shape='polygon')

        super(RegularOperator, self).__init__(vertices, **kwargs)

    def get_patch(self, *args, **kwargs):
            return self.get_polygon(*args, **kwargs)


class RegularBubbleOperator(RegularOperator):
    """
    This operator is represented by a circle with N vertices.
    """
    def __init__(self, N, center, size=0.3, angle=None, **kwargs):
        self.radius = size
        super(RegularBubbleOperator, self).__init__(N, center, size, angle, **kwargs)

    def get_circle(self, *args, **kwargs):
        return mpa.Circle(self.get_center(), self.radius, **self.style)

    def get_patch(self, *args, **kwargs):
            return self.get_circle(*args, **kwargs)
