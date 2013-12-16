
from copy import deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

# Personnal modules
import mycolors as mc

from .. import vectors
from .. import colors
from ..constants import tau



class Operator(object):
    """
    A N-point operator.
    Often represented as a polygon, or a circle.


    Arguments
    ---------

    verticles : feynman.Verticle (=2*[Verticle()])
        First and second verticle, counted clockwise
        defining an edge (or the starting and ending pointaa)
        of a patch object. 

    N : (2)
        Number of verticles to the operator.

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

    shape :
        default     -   'oval' if N == 2
                    -   'polygon' if N > 2.

    verticles :

    N :

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

        if self.N == 2:
            self.shape = 'ellipse'
        else:
            self.shape = 'polygon'

        self.ellipse_excentricity = kwargs.pop('c')

        self.style = dict(
            edgecolor="k",
            facecolor=colors.grey,
            linewidth=3,
            )

        self.style.update(kwargs)

        self.texts = list()

    def get_verticles(self):
        """Return the verticles."""
        return self.verticles

    def set_verticles(self, *verticles):
        """Return the verticles."""
        self.verticles = verticles

    def get_xy(self):
        """Return the xy coordinates of the verticles, clockwise."""
        return np.array([v.xy for v in self.verticles])

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
        else:
            raise ValueError("Unrecognized shape: " + self.shape)

    def get_polygon(self):
        """Return the polygon."""
        polygon = mpa.Polygon(self.get_xy(), **self.style)
        return polygon

    def get_ellipse(self):
        """Return the oval."""
        start, end = self.get_xy()
        dxy = end - start
        width = np.linalg.norm(dxy)
        height = width / self.ellipse_excentricity
        center = self.get_center()
        angle = vectors.angle(dxy, 'deg')
        ellipse = mpa.Ellipse(center, width, height, angle=angle, **self.style)
        return ellipse

    def text(self, s, x=-.025, y=-.025, **kwargs):
        """
        Add text in the operator.

        Arguments
        ---------

        s : Text string.

        x : (-0.025)
            x position, relative to the center of the operator.

        y : (-0.025)
            y position, relative to the center of the operator.

        fontsize : (28)
            The font size.

        **kwargs :
            Any other style specification for a matplotlib.text.Text instance.
"""
        default = dict(fontsize=28)
        for key, val in default.items():
            kwargs.setdefault(key, val)
        self.texts.append((s, x, y, kwargs))

    def get_texts(self):
        """Return a list of matplotlib.text.Text instances."""
        texts = list()
        for (s, x, y, kwargs) in self.texts:
            center = self.get_center()
            xtext, ytext = center + np.array([x,y])
            texts.append(mpt.Text(xtext, ytext, s, **kwargs))
        return texts

    def draw(self, ax):
        """Draw the diagram."""
        patch = self.get_patch()
        ax.add_patch(patch)
        for text in self.get_texts():
            ax.add_artist(text)
        return


