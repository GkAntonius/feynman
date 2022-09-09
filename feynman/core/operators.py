
from copy import deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

from . import Drawable
from . import Line, Vertex

from .. import vectors
from .. import colors
from ..constants import tau

__all__ = ['Operator', 'RegularOperator', 'RegularBubbleOperator']

class Operator(Drawable):
    """
    A N-point operator.
    Often represented as a polygon, or a circle.


    Parameters
    ----------

    vertices: a list of N vertices (feynman.Vertex)
        First and second vertex, counted clockwise
        defining an edge (or the starting and ending points)
        of a patch object. 

    rotate: (0.)
        Rotation angle to the operator, in units of tau.

    c: (1.)
        If the shape is elliptic, c is the excentricity of the ellipse,
        that is, the ratio of the long axe over the short axe.
        When c = 1, the shape will be a circle.

    shape:
        By default, the shape is 'ellipse' if N=2,
        and 'polygon' if N>2.
        When N=4 however, you can also specify 'bubble' as a shape
        to have lines connecting (v1, v2) together and (v3, v4) together,
        and a single elliptic patch on top of those two lines.

    line_prop:
        Line properties if shape='bubble'.

    **kwargs:
        Any other style specification for a matplotlib.patches.Patch instance.

    """

    _vertices = list()
    _style = dict()

    def __init__(self, vertices, **kwargs):
        super(Operator, self).__init__()

        # Default values
        default = dict(
            c=1,
            )

        # Set default values
        for key, val in default.items():
            kwargs.setdefault(key, val)

        self.vertices = vertices

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

        self.style = dict(
            edgecolor="k",
            facecolor=colors.lightblue,
            linewidth=3,
            )

        self.style.update(kwargs)

        self.line_prop = kwargs.pop('line_prop', dict())

        default_line_prop = dict(
            style = 'simple straight linear',
            arrow = False,
            zorder = 1,
            linewidth=3,
            )

        for key, val in default_line_prop.items():
            self.line_prop.setdefault(key, val)

        self.texts = list()
        self.lines = list()

    @property
    def N(self):
        """Number of vertices to the operator."""
        return len(self.vertices)

    @property
    def vertices(self):
        """The list of N vertices forming the operator."""
        return self._vertices

    @vertices.setter
    def vertices(self, vlist):
        self._vertices = vlist

    @property
    def style(self):
        """
        A dictionary of style elements for a :class:`matplotlib.patches.Patch`
        instance, such as linewidth, edgecolor, facecolor, etc.
        """
        return self._style

    @style.setter
    def style(self, dictionary):
        self._style = dictionary

    def set_angles(self, *angles):
        """Set the angles between vertices."""
        raise NotImplementedError()

    def set_center(self, xy):
        """Set the center of the polygon """
        raise NotImplementedError()

    def get_xy(self):
        """Return the xy coordinates of the vertices, clockwise."""
        return np.array([v.xy for v in self.vertices])

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

        v1, v2, v3 ,v4 = self.vertices
        line1 = Line(v1, v2, **self.line_prop)
        line2 = Line(v3, v4, **self.line_prop)

        return line1, line2

    def text(self, s, x=0., y=0., **kwargs):
        """
        Add text in the operator.

        Parameters
        ----------

        s: Text string.

        x: (0.)
            x position, relative to the center of the operator.

        y: (0.)
            y position, relative to the center of the operator.

        fontsize: (30)
            The font size.

        **kwargs :
            Any other style specification for a matplotlib.text.Text instance.
        """
        default = dict(fontsize=30, zorder=10, ha='center', va='center')
        text_kwargs = deepcopy(kwargs)
        for key, val in default.items():
            text_kwargs.setdefault(key, val)
        self.texts.append([s, x, y, text_kwargs])

    def get_texts(self):
        """Return a list of matplotlib.text.Text instances."""
        texts = list()
        for (s, x, y, kwargs) in self.texts:
            center = self.get_center()
            xtext, ytext = center + np.array([x,y])
            if self.diagram._draggable:
                # Allow drag and drop, if not explicitly disabled
                if not "picker" in kwargs: kwargs["picker"] = True
            texts.append(mpt.Text(xtext, ytext, s, **kwargs))
        return texts

    def _update_text_position(self, i, dx, dy):
        s, x, y, kwargs = self.texts[i]
        self.texts[i] = (s, x + dx, y + dy, kwargs)

    def draw(self, ax):
        """Draw the diagram."""
        for line in self.get_lines():
            line.draw(ax)
        patch = self.get_patch()
        ax.add_patch(patch)
        for i, text in enumerate(self.get_texts()):
            ax.add_artist(text)
            if self.diagram._draggable:
                self.diagram._artists[text] = (self, i)
        return

    def scale_width(self, x):
        """Apply a scaling factor to the line width."""
        self.style['linewidth'] *= x
        self.line_prop['linewidth'] *= x

    def scale_text(self, x):
        """Apply a scaling factor to the text size and relative position."""
        for textparams in self.texts:
            textparams[1] *= x
            textparams[2] *= x
            textparams[3]['fontsize'] *= x

    def scale(self, x):
        """Apply a scaling factor."""
        self.scale_width(x)
        self.scale_text(x)


class RegularOperator(Operator):
    """
    A N-point operator represented as a polygon.

    Parameters
    ----------

    N:
        Number of vertices.

    center:
        Position of the center of the operator.

    size:
        Distance from the center to a corner.

    angle:
        vertices are counted clockwise. angle=0=1 means
        that there is a vertex in the [1,0] direction from the center.

    **kwargs :
        Any other style specification for a matplotlib.patches.Patch instance.
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
            v = Vertex(xy=center, radius=size, angle=angle + float(i)/N)
            vertices.append(v)

        kwargs.update(shape='polygon')

        super(RegularOperator, self).__init__(vertices, **kwargs)

    def get_patch(self, *args, **kwargs):
            return self.get_polygon(*args, **kwargs)


class RegularBubbleOperator(RegularOperator):
    """
    This operator is represented by a circle with N vertices.

    Parameters
    ----------

    N:
        Number of vertices.

    center:
        Position of the center of the operator.

    size:
        Radius of the circle.

    angle:
        vertices are counted clockwise. angle=0=1 means
        that there is a vertex in the [1,0] direction from the center.

    **kwargs :
        Any other style specification for a matplotlib.patches.Patch instance.
    """
    def __init__(self, N, center, size=0.3, angle=None, **kwargs):
        self.radius = size
        super(RegularBubbleOperator, self).__init__(N, center, size, angle, **kwargs)

    def get_circle(self, *args, **kwargs):
        return mpa.Circle(self.get_center(), self.radius, **self.style)

    def get_patch(self, *args, **kwargs):
            return self.get_circle(*args, **kwargs)
