#
# TODO
# ----
#
#   o Verticle
#       - get_lines
#
#   o Operator
#
#   o Diagram
#       - add_text
#       - verticles (add multiple verticles at once)
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

# =========================================================================== #


class Verticle(object):
    """
    A verticle. Often represented as a point.

    Arguments
    ---------

    xy :
        Coordinates.

    **kwargs :
        Any matplotlib line style argument. 
"""

    _xy = np.zeros(2)

    def __init__(self, xy, **kwargs):

        self.xy = xy

        self.style = dict(
            marker='o',
            linestyle='',
            markersize=10,
            color='k',
            zorder=20,
            )

        self.style.update(kwargs)

        # TODO Should be able to get the lines connected to that verticle.
        self.lines = list()
        self.texts = list()

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

    def text(self, s, x=-.025, y=-.025, **kwargs):
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
        default = dict(fontsize=14)
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


# =========================================================================== #


class Line(object):
    """
    A particle line joigning two verticles.

    Arguments
    ---------

    vstart :
        Starting vericle.

    vend :
        End vericle.

    pathtype : ( linear)
        The shape of the line path.

            linear    -  A straight line between two points.
            elliptic  -  An ellipse arc.
            circular  -  A circle starting and ending at the same verticle.

    linetype : ( simple )
        The type of line.

            simple  -  A straight line.
            wiggly  -  A wavy line.
            loopy   -  A spring.

    linestyle : ( single )
        The style for the line.

            single  -  A simple line.
            double  -  A double line.

    ellipse_spread : (0.5)
        The angle (in units of 2pi) spread by the ellipse arc.
        The limit cases are
            0.0 --> the curve will tend to a straight ligne,
            0.5 --> the curve will be half an ellipse,
            1.0 --> the curve will tend to a closed ellipse. 

    ellipse_exc : (1.2)
        The excentricity of the ellipse, that is, the ratio of the long axe
        over the short axe. At 1.0, the curve will be a circle arc.
        Also Controls wether the curve is 'up' or 'down'.
        A positive value makes it 'up', while a negative value makes it 'down'.

    circle_radius : float (.1)
        The radius of the circle.

    circle_angle : float (0.25)
        The angle of the anchor verticle to the circle center, in units of 2pi.

    arrow : bool ( True )
        Include an arrow in the line.

    arrowparam : dict
        In case arrow==True, gives a mapping of all parameters for add_arrow.

    nwiggles : float
        The number of wiggles in a wiggly line.
        Can be integer or half-integer (if the phase is 0 or .5).

    nloops : float
        The number of loops in a loopy line.

    phase : float
        Phase in the wiggly or loopy pattern, in units of 2pi.

    npoints : int
        Number of points that makes up the line.


    Properties
    ----------

    npoints : int
        Number of points that makes up the line.

    xy : np.ndarray, shape (npoints, 2)
        The xy coordinates of the line.

    linepath : np.ndarray, shape (npoints, 2)
        The director line arount which the xy points are placed (with style).

    tangent : np.ndarray, shape (npoints, 2)
        The unitary vectors tangent to the linepath.

    normal : np.ndarray, shape (npoints, 2)
        The unitary vectors normal to the linepath.

"""

    _xy = np.zeros((2, 2))
    _linepath = np.zeros((2, 2))
    _tangent = np.zeros((2,2))
    _normal = np.zeros((2,2))
    _main_lines = None
    t =  np.linspace(0, 1, 2)

    def __init__(self, vstart, vend, **kwargs):

        self.vstart = vstart
        self.vend = vend

        self.xstart, self.ystart = self.vstart.xy
        self.xend, self.yend = self.vend.xy

        # Default values
        default = dict(
            linestyle='single',
            linetype='simple',
            pathtype='linear',
            arrow=False,
            npoints=400,
            ellipse_spread=.5,
            ellipse_exc=1.2,
            circle_radius=.1,
            circle_angle=0.25,
            amplitude=.025,
            xamp=.025,
            yamp=.05,
            nwiggles=5,
            nloops=14,
            phase=0,
            )

        # Adjust some default values according to pathtype and linetype
        if (kwargs.get('pathtype') == 'circular' and
            kwargs.get('linetype') == 'wiggly'):
            default.update(nwiggles=7)
            default.update(phase=.25)

        # Set default values
        for key, val in default.items():
            kwargs.setdefault(key, val)

        # Set attributes values
        for key in (

            'arrow',

            'ellipse_spread',
            'ellipse_exc',
            'circle_radius',
            'circle_angle',

            # Amplitude of the wave and such...
            'amplitude',
            'xamp',
            'yamp',

            # Wiggly and loopy line parameters
            'nwiggles',
            'nloops',
            'phase',

            ):
            self.__dict__[key] = kwargs.pop(key)

        # Set the line type
        self.set_pathtype(kwargs.pop('pathtype'))
        self.set_linetype(kwargs.pop('linetype'))
        self.set_linestyle(kwargs.pop('linestyle'))

        arrowparam = kwargs.pop('arrowparam', dict())

        # Arrows parameters
        self.arrows_param = list()

        self.lines = list()
        self.patches = list()
        self.texts = list()

        if self.arrow:
            self.add_arrow(**arrowparam)

        # Set the main paramter
        self._set_t(kwargs.pop('npoints'))

        # Compute the linepath
        self._set_linepath()

        # Compute the tangent
        self._set_tangent()

        # Compute the normal
        self._set_normal()

        # Compute the line points
        self._set_xy()

        # all other kwargs are 'matplotlib' line style arguments
        self.style = dict(
            marker='',
            color='k',
            linestyle='-',
            linewidth=3,
            zorder=10,
            )
        self.style.update(kwargs)
        self.double_center_color = 'w'


    def _set_t(self, npoints):
        """Set the main parameter for the curve."""
        self.t = np.linspace(0, 1, npoints)

    # Morphable
    def _set_linepath(self):
        """Compute the line path."""
        return self._set_linear_linepath()

    @property
    def rstart(self): return self.vstart.xy

    @property
    def rend(self): return self.vend.xy

    @property
    def dr(self): return self.rend - self.rstart

    @property
    def angle(self):
        """The angle (units of 2pi) of the direct line between end points."""
        dx, dy = self.dr
        angle = np.arctan(dy / dx)
        if dx < 0: angle += np.pi
        return angle / (2 * np.pi)

    @property
    def xy(self):
        return self._xy

    @xy.setter
    def xy(self, xy):
        self._xy = np.array(xy)
        assert self.xy.ndim == 2, "Wrong dimension for line xy."
        assert self.xy.shape[1] == 2, "Wrong dimension for line xy."

    @property
    def npoints(self):
        return self.t.size

    @npoints.setter
    def npoints(self, n):
        self._set_t(n)

    def _set_tangent(self):
        """Compute the tangent points."""
        self._set_tangent_numeric()

    @property
    def tangent(self):
        return self._tangent

    @tangent.setter
    def tangent(self, xy):
        self._tangent = np.array(xy)

    def _set_tangent_numeric(self):
        """Compute tangent numerically."""
        v = np.zeros((self.npoints, 2))
        v[:-1] = self.xy[1:] - self.xy[:-1]
        v[-1] = self.xy[-1] - self.xy[-2]
        norms = np.array(map(lambda n: max(np.linalg.norm(n), 1e-8), v))
        self.tangent = vectors.dot(v, 1. / norms)

    @property
    def normal(self):
        return self._normal

    def _set_normal(self):
        """Compute the normal vector along the curve."""
        R = np.array([[0., -1.],[1., 0.]])
        self._normal = vectors.sqdot(self.tangent, R)

    def set_pathtype(self, pathtype):
        """
        Set the path type.

            linear    -  A straight line between two points.
            elliptic  -  An ellipse arc.
            circular  -  A circle starting and ending at the same verticle.
"""
        if pathtype in ('linear', 'l', 'straight'):
            self._set_linepath = self._set_linear_linepath
        elif pathtype in ('elliptic', 'e', 'ellipse'):
            self._set_linepath = self._set_elliptic_linepath
        elif pathtype in ('circular', 'c', 'circle'):
            self._set_linepath = self._set_circular_linepath
        else:
            raise ValueError('Wrong value for pathtype')

    # User
    def get_xy(self):
        """Return the xy array of the line points, of shape (numpoins, 2)."""
        return self.xy

    # User
    def set_xy(self, xy):
        """Set the xy array of the line points, of shape (numpoins, 2)."""
        self.xy = xy


    #   -------------------------------------------------------------------   #

    def add_arrow(self, t=0.5, direction=1, theta=.083, size=.025, **kwargs):
        """
        Add an arrow on the line.

        Arguments
        ---------

        t : float
            The position of the arrow along the line.
            Must be in the range [0,1].

        direction :
            The direction of the arrow. A positive number gives
            a forward arrow while a negative number gives a backward arrow.

        theta :
            The angle the arrow branches make with the path
            in units of 2pi

        size :
            The length of the arrow branches.

        **kwargs :
            Any style specification, such as linewidth.
"""
        # TODO: arrow style to allow a full triangle shape
        if not (t >= 0 and t <= 1):
            raise ValueError("t should be in range [0,1]")
        param = (t, direction, theta, size, kwargs)
        self.arrows_param.append(param)

    def text(self, s, t=.5, y=-.06, **kwargs):
        """
        Add text over the line.

        Arguments
        ---------

        s : Text string.

        t : (.5)
            Position along the line (0 < t < 1).

        y : (-.06)
            y position, perpendicular to the path direction.

        fontsize : (14)
            The font size.

        **kwargs :
            Any other style specification for a matplotlib.text.Text instance.
"""
        default = dict(fontsize=14)
        for key, val in default.items():
            kwargs.setdefault(key, val)
        self.texts.append((s, t, y, kwargs))

    def get_texts(self):
        """Return a list of matplotlib.text.Text instances."""
        texts = list()
        for textparams in self.texts:
            (s, t, y, kwargs) = textparams
            middle = self.get_path_point(t)
            normal = self.get_normal_point(t)
            xtext, ytext = middle + y * normal
            texts.append(mpt.Text(xtext, ytext, s, **kwargs))
        return texts

    def get_arrow_lines(self):
        """Get the arrow lines."""
        linepath = self.get_linepath()

        lines = list()
        for param in self.arrows_param:
            (t, d, theta, size, style) = param

            th = 2 * np.pi * theta

            # Index of t
            it = int(t * self.npoints)

            # Find the position and tangent vector at point t
            rtip = linepath[it]
            tan = self.tangent[it]
            norm = self.normal[it]

            # Starting point
            drstart = - d * np.cos(th) * tan + np.sin(th) * norm
            rstart = rtip + size * drstart

            # End point
            drend = - d * np.cos(th) * tan - np.sin(th) * norm
            rend = rtip + size * drend

            # Set default style
            for key in ('linewidth', 'color', 'linestyle', 'marker'):
                style.setdefault(key, self.style[key])
            style.setdefault('zorder', self.style.get('zorder') + 2)

            xy = np.array([rstart, rtip, rend]).transpose()
            arrow_line = mpl.lines.Line2D(*xy, **style)

            lines.append(arrow_line)

        return lines

    @property
    def main_lines(self):
        return self._main_lines

    def set_linestyle(self, linestyle):
        """
        Set the linestyle.

            single  -  A simple line.
            double  -  A double line.
"""
        if linestyle in ('simple', 's', 'single'):
            self.get_main_lines = self.get_single_main_lines
        elif linestyle in ('double', 'd'):
            self.get_main_lines = self.get_double_main_lines
        else:
            raise ValueError('Wrong value for linestyle')

    # Morphable
    def _set_main_lines(self):
        """Set the main lines."""
        self._set_single_main_line()

    # Morphable
    def get_main_lines(self):
        """Get the main lines."""
        return self.get_single_main_lines()

    def get_double_main_lines(self):
        """Get a set of lines forming a double line."""
        lines = list()
        style = deepcopy(self.style)
        x, y = self.xy.transpose()

        # Make contour lines
        style['zorder'] = self.style.get('zorder', 0) -1
        style['linewidth'] = 1.8 * self.style.get('linewidth', 2)
        lines.append(mpl.lines.Line2D(x, y, **style))

        # Make center lines
        style['zorder'] = self.style.get('zorder', 0)
        style['color'] = self.double_center_color
        style['linewidth'] = .5 * self.style.get('linewidth', 2)
        lines.append(mpl.lines.Line2D(x, y, **style))
        return lines
    def _set_double_main_lines(self): self.main_lines = get_double_main_lines()

    def get_single_main_lines(self):
        """Get the main lines."""
        line = mpl.lines.Line2D(*self.xy.transpose(), **self.style)
        #self.main_lines = [line]
        return [line]
    def _set_single_main_lines(self): self.main_lines = get_single_main_lines()
        
    def distance(self):
        """The distance between the starting point and the end point."""
        return np.linalg.norm(self.dr)

    def pathlength(self, linepath=None):
        """The length of the path."""
        if linepath is None:
            linepath = self.get_linepath()

        # Integrate the linepath
        dl = linepath[1:] - linepath[:-1]
        norms = np.sqrt(sum(dl.transpose() * dl.transpose()))
        l = sum(norms)
        return l

    def linelength(self, linepath=None):
        """The length of the path."""
        if line is None:
            line = self.get_line()

        # Integrate the line
        dl = line[1:] - line[:-1]
        norms = np.sqrt(sum(dl.transpose() * dl.transpose()))
        l = sum(norms)

        return l

    @property
    def linepath(self):
        return self._linepath

    @linepath.setter
    def linepath(self, linepath):
        self._linepath = np.array(linepath)
        assert self.linepath.ndim == 2, "Wrong dimension for line path."
        assert self.linepath.shape[1] == 2, "Wrong dimension for line path."
        self._set_tangent()
        self._set_normal()
        self._set_xy()

    # User
    def get_linepath(self, *args, **kwargs):
        """
        Get the director line arount which the line exhibits style.

        Returns
        -------
        xy: np.ndarray of shape (N, 2)

"""
        return self.linepath

    def get_path_point(self, t):
        """
        Get xy vector for a particular position along the path.

        t: Distance parameter along the path. 0. <= t <= 1.

        Returns: np.ndarray of shape (2)
"""
        # This function is not optimized at all.
        linepath = self.get_linepath()
        i_xy = min(int(t * self.npoints), self.npoints)
        xy = linepath[i_xy]
        return xy.reshape(2)
        
    def _set_linear_linepath(self):
        """Compute the xy eectors for a linear line path."""
        v, t = np.meshgrid(self.dr, self.t)
        self.linepath = self.rstart + v * t
        
    def _set_elliptic_linepath(self):
        """Compute the xy vectors for an elliptic line path."""

        # Geometry of the starting and end points
        l = self.distance()

        # Rotation matrix.
        gamma = self.angle
        R = np.array([[np.cos(gamma), - np.sin(gamma)],
                      [np.sin(gamma),   np.cos(gamma)]])

        # Ellipse center
        ro = self.rstart + self.dr / 2

        # Axes of the ellipse
        a = l / (2 * np.sin(self.ellipse_spread * np.pi))
        b = a / self.ellipse_exc

        # Angular progression along the ellipse.
        theta_s = np.pi * (1 - 2 * self.ellipse_spread) / 2.
        theta_i = 2 * np.pi * self.ellipse_spread
        theta = theta_s + theta_i * self.t

        # xy relative to the ellipse center
        ellipse = np.array([-a*np.cos(theta), b*np.sin(theta)]).transpose()

        # rotate ellipse and shift vector
        ellipse = vectors.sqdot(ellipse, R)
        self.linepath = vectors.add(ellipse, ro)

    def _set_circular_linepath(self):
        """Get xy vectors for the path."""

        r = self.circle_radius
        alpha = self.circle_angle

        # Circle center
        ro = self.rend + vectors.rotate([r,0], alpha)

        # Angular progression along the circle.
        theta = 2 * np.pi * (self.t + alpha)

        # xy relative to the circle center
        circle = r * np.array([- np.cos(theta), - np.sin(theta)]).transpose()

        # shift vector
        self.linepath = vectors.add(circle,  ro)

    # User
    def get_tangent(self):
        """
        Get xy vectors for the tangent path.
        These vectors are normalized.

        Otional arguments
        -----------------
        linepath :
            Line path on which to compute. Compute the path if None.

        Returns
        -------
            np.ndarray of shape (N, 2)
"""
        return self.tangent

    def get_tangent_point(self, t):
        """
        Get a particular tangent vector for a point along the path.

        t: Distance parameter along the path. 0. <= t <= 1.

        Returns: np.ndarray of shape (2)
"""
        i_v = min(int(t * self.npoints), self.npoints)
        return self.tangent[i_v].reshape(2)

    # User
    def get_normal(self, tangent=None, **kwargs):
        """
        Return a vector normal to the path for each parameter point.
        The vectors are normalized.

        Otional arguments
        -----------------
        tangent :
            Line path on which to compute. Compute the path if None.

        Returns
        -------
            np.ndarray of shape (N, 2)
"""
        return self.normal

    def get_normal_point(self, t):
        """
        Get a particular normal vector for a point along the path.

        t: Distance parameter along the path. 0. <= t <= 1.

        Returns: np.ndarray of shape (2)
"""
        i_v = min(int(t * self.npoints), self.npoints)
        v = self.normal[i_v]
        return v.reshape(2)

    # Morphable
    def _set_xy(self):
        """Compute the xy coordinates of the line."""
        return self._set_xy_simple()

    def set_linetype(self, linetype):
        """
        Set the line style.

            simple  -  A straight line.
            wiggly  -  A wavy line.
            loopy   -  A spring.
"""
        if linetype in ('simple', 's', 'straight'):
            self._set_xy = self._set_xy_simple
        elif linetype in ('wiggly', 'w', 'wiggle'):
            self._set_xy = self._set_xy_wiggly
        elif linetype in ('loopy', 'l', 'loop'):
            self._set_xy = self._set_xy_loopy
        else:
            raise ValueError('Wrong value for linetype')

    def _set_xy_simple(self):
        """Compute the xy simple path.""" 
        self.xy = self.linepath

    def _set_xy_wiggly(self):
        """Compute the xy wiggly path.""" 

        numhalfwaves = int(2 * self.nwiggles)

        omega = np.pi * numhalfwaves
        phi = 2 * np.pi * self.phase

        sine = np.sin(omega * self.t + phi)

        dxy = self.amplitude * vectors.dot(self.normal, sine)
        dxy = vectors.add(dxy, -dxy[0])

        self.xy = self.linepath + dxy

    def _set_xy_loopy(self):
        """Compute the xy wiggly path.""" 

        omega = 2 * np.pi * self.nloops
        phi = 2 * np.pi * self.phase

        dy = - np.cos(omega * self.t + phi)
        dy -= dy[0]

        dx = np.sin(omega * self.t + phi)
        dx -= dx[0]

        dxy = (self.xamp * vectors.dot(self.tangent, dx) +
               self.yamp * vectors.dot(self.normal, dy))

        self.xy = self.linepath + dxy

    def get_lines(self):
        """Get the lines."""  # Could be faster
        lines = list()
        lines.extend(self.get_main_lines())
        lines.extend(self.get_arrow_lines())
        return lines

    def draw(self, ax):
        """Plot the line."""
        for line in self.get_lines():
            ax.add_line(line)
        for text in self.get_texts():
            ax.add_artist(text)
        return

# =========================================================================== #


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
        Rotation angle to the operator, in units of 2pi.

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

        self.ellipse_exc = kwargs.pop('c')

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
        height = width / self.ellipse_exc
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


# =========================================================================== #


class Diagram(object):
    """
    The main object for a feynman diagram.

    Arguments
    ---------

    fig : A matplotlib Figure. If none is given, a new one is initialized.
    ax : A matplotlib AxesSubplot. If none is given, a new one is initialized.
"""
    def __init__(self, ax=None):

        if ax is not None:
            self.ax = ax
        else:
            fig = plt.figure(figsize=(6,6))
            self.ax = fig.gca()
            self.ax.set_xlim(0,1)
            self.ax.set_ylim(0,1)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        self.verticles = list()
        self.lines = list()
        self.operators = list()

    def verticle(self, xy=(0,0), **kwargs):
        """
        Create a verticle.

        Arguments
        ---------

        xy :
            Coordinates.

        **kwargs :
            Any matplotlib line style argument. 

        Returns
        -------

        feynman.Verticle instance.
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

        list of feynman.Verticle instance.
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
        """Create a line."""
        l = Line(*args, **kwargs)
        self.lines.append(l)
        return l

    def operator(self, *args, **kwargs):
        """Create an operator."""
        O = Operator(*args, **kwargs)
        self.operators.append(O)
        return O

    def add_verticle(self, verticle):
        """Add a verticle."""
        self.verticles.append(verticle)

    def add_line(self, line):
        """Add a line."""
        self.lines.append(line)

    def add_operator(self, operator):
        """Add an operator."""
        self.operators.append(operator)

    def plot(self):
        """Draw the diagram."""

        for v in self.verticles:
            v.draw(self.ax)

        for l in self.lines:
            l.draw(self.ax)

        for O in self.operators:
            O.draw(self.ax)

    def text(*args, **kwargs):
        """Add text using matplotlib.axes.Axes.text."""
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
