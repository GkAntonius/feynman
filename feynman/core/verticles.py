
from copy import deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

from . import Drawable

from .. import vectors
from .. import colors
from ..constants import tau


class Verticle(Drawable):
    """
    A verticle. Often represented as a point.

    Arguments
    ---------

    xy :
        Coordinates.

    dxy :
        Coordinates, so that the position is given by xy + dxy.

    dx :
        Coordinate, so that the position is given by xy + [dx, 0].

    dy :
        Coordinate, so that the position is given by xy + [0, dy].

    angle :
        Angle from xy, so that the position is given by
        xy + radius * [cos(angle), sin(angle)]
        Angle is given in units of tau=2pi

    radius :
        Radius from xy, so that the position is given by
        xy + radius * [cos(angle), sin(angle)]

    **kwargs :
        Any matplotlib line style argument. 
    """

    _xy = np.zeros(2)

    _lines = list()

    def __init__(self, xy=(0,0), **kwargs):

        dx = np.array(kwargs.pop('dx', 0.))
        dy = np.array(kwargs.pop('dy', 0.))
        dxy = np.array(kwargs.pop('dxy', (0.,0.)))
        angle = np.array(kwargs.pop('angle', 0.))
        radius = np.array(kwargs.pop('radius', 0.))

        self.xy = ( xy 
            + dxy + np.array([dx, dy])
            + radius * np.array([np.cos(angle*tau), np.sin(angle*tau)])
            )

        self.style = dict(
            marker='o',
            linestyle='',
            markersize=10,
            color='k',
            zorder=20,
            )

        self.style.update(kwargs)

        # TODO Should be able to get the lines connected to that verticle.
        self.texts = list()

    @property
    def x(self): return self._xy[0]

    @x.setter
    def x(self, val): self._xy[0] = val

    @property
    def y(self): return self._xy[1]

    @y.setter
    def y(self, val): self._xy[1] = val

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, xy):
        self._xy = np.array(xy)
        assert self.xy.ndim == 1, "Wrong dimension for line xy."
        assert self.xy.size == 2, "Wrong dimension for line xy."

    # User
    def set_xy(self, xy):
        self.xy = xy

    def get_marker(self):
        """Returns a matplotlib.lines.Line2D instance."""
        return mpl.lines.Line2D([self.xy[0]],[self.xy[1]], **self.style)

    # TODO
    # Change x, y for dx, dy
    def text(self, s, x=-.025, y=+.025, **kwargs):
        """
        Add text near the verticle.

        Arguments
        ---------

        s : Text string.

        x : (-0.025)
            x position, relative to the verticle.

        y : (-0.025)
            y position, relative to the verticle.

        fontsize : (14)
            The font size.

        **kwargs :
            Any other style specification for a matplotlib.text.Text instance.
        """
        default = dict(
            verticalalignment='center',
            horizontalalignment='center',
            fontsize=14
            )
        for key, val in default.items():
            kwargs.setdefault(key, val)
        self.texts.append((s, x, y, kwargs))

    def get_texts(self):
        """Return a list of matplotlib.text.Text instances."""
        texts = list()
        for (s, x, y, kwargs) in self.texts:
            xtext, ytext = self.xy + np.array([x,y])
            texts.append(mpt.Text(xtext, ytext, s, **kwargs))
        return texts

    def draw(self, ax):
        marker = self.get_marker()
        ax.add_line(marker)
        for text in self.get_texts():
            ax.add_artist(text)
        return

    @property
    def lines(self):
        """The lines attached to it."""
        return self._lines

    @lines.setter
    def lines(self, value):
        self._lines = value

    def chunk(self, *args, **kwargs):
        self.diagram.add_chunk(self, *args, **kwargs)

    #@classmethod
    #def _add_relative(cls, other, radius, angle):
    #    """
    #    Return a new instance relative to some other existing verticle.
    #    """


# =========================================================================== #

