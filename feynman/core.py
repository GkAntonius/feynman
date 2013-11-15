#
# TODO
# ----
#
#   o Verticle
#       - get_lines
#
#   o Operator
#       - Text
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
    def __init__(self, xy=(0,0), **kwargs):

        self.xy = np.array(xy)

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

    linestyle :
        This is just the style of the line.
            'simple'
            'wiggly'
            'double'
            'doublewiggly'
            'loopy'
            'doubleloopy'
            'dotted'
            'doubledotted'
            'dashed'
            'doubledashed'

    pathtype :
        The shape of the line path.

            linear    -  A straight line between two points.
            elliptic  -  An ellipse arc.
            circular  -  A circle starting and ending at the same verticle.

    ellipseparam : (float, float, int) = (.5, 1.2, 1)
        Three parameters governing the shape of an elliptic path:

            alpha  -  The angle (in units of 2pi) spread by the ellipse arc.
                      When alpha --> 0 the curve will tend to a straight ligne,
                      when alpha = 0.5 the curve will be half an ellipse,
                      when alpha --> 1 the curve will tend to a closed ellipse. 

            c      -  The excentricity of the ellipse, that is, the ratio
                      of the long axe over the short axe.
                      When c = 1, the curve will be a circle arc.

            d      -  Controls wether the curve is 'up' or 'down'.
                      A positive value makes it 'up', while a negative value
                      makes it 'down'.

    circleparam : (float, int) = (.25, 1)
        Two parameters governing the shape of a circular path: (r, d).

            r      -  The radius of the circle.

            d      -  Controls wether the curve is 'up' or 'down'.
                      A positive value makes it 'up', while a negative value
                      makes it 'down'.

    arrow : bool = True
        Include an arrow in the line.

    arrow_param : dict
        In case arrow==True, gives a mapping of all parameters for add_arrow.

    nwiggles : float
        The number of wiggles in a wiggly line.
        Can be integer or half-integer (if the phase is 0 or .5).

    nloops : float
        The number of loops in a loopy line.

    phase : float
        Phase in the wiggly or loopy pattern, in units of 2pi.


"""
    def __init__(self, vstart, vend, **kwargs):

        self.vstart = vstart
        self.vend = vend

        self.xstart, self.ystart = self.vstart.xy
        self.xend, self.yend = self.vend.xy

        # Default values
        default = dict(
            linestyle='simple',
            pathtype='linear',
            arrow=False,
            numpoints=400,
            ellipseparam=(.5, 1.2, 1),
            circleparam=(.25, 1),
            amplitude=.025,
            xamp=.025,
            yamp=.05,
            nwiggles=6,
            nloops=14,
            phase=0,
            )

        # Adjust some default values according to pathtype and linestyle
        if (kwargs.get('pathtype') == 'circular' and
            kwargs.get('linestyle') == 'wiggly'):
            default.update(nwiggles=7)
            default.update(phase=.25)


        # Set default values
        for key, val in default.items():
            kwargs.setdefault(key, val)

        # Set attributes values
        for key in (

            'linestyle',
            'pathtype',
            'arrow',

            # number of points for the line
            'numpoints',
            'ellipseparam',
            'circleparam',

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

        arrow_param = kwargs.pop('arrow_param', dict())

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


        # Main parameter for the curve
        self.t = np.linspace(0, 1, self.numpoints)

        # Elliptic parameters.
        self.alpha, self.c, self.d = self.ellipseparam

        # Circular parameters.
        self.r, self.d = self.circleparam  # TODO avoid redundancy

        # Arrows parameters
        self.arrows_param = list()

        self.lines = list()
        self.patches = list()
        self.texts = list()

        if self.arrow:
            self.add_arrow(**arrow_param)

    @property
    def rstart(self):
        """The starting position."""
        return np.array(self.vstart.xy)

    @property
    def rend(self):
        """The end position."""
        return np.array(self.vend.xy)

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
        if not (t >= 0 and t <= 1):
            raise ValueError("t should be in range [0,1]")
        param = (t, direction, theta, size, kwargs)
        self.arrows_param.append(param)

    def text(self, s, t=.5, y=.1, **kwargs):
        """
        Add text over the line.

        Arguments
        ---------

        s : Text string.

        t : (.5)
            Position along the line (0 < t < 1).

        y : (.1)
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
        tangent = self.get_tangent(linepath)
        normal = self.get_normal(tangent)

        lines = list()
        for param in self.arrows_param:
            (t, d, theta, size, style) = param

            th = 2 * np.pi * theta

            # Index of t
            it = int(t * self.numpoints)

            # Find the position and tangent vector at point t
            rtip = linepath[it]
            tan = tangent[it]
            norm = normal[it]

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


    def get_main_lines(self):
        """Get the main lines."""
        if 'double' in self.linestyle:
            return self.get_double_main_lines()
        else:
            return self.get_single_main_lines()
        #if self.linestyle in ('simple', 's'):
        #    return self.get_simple_main_lines()
        #elif self.linestyle in ('wiggly', 'w'):
        #    return self.get_wiggly_main_lines()
        #elif self.linestyle in ('loopy', 'l'):
        #    return self.get_loopy_main_lines()
        #else:
        #    raise ValueError('Wrong value for linestyle.')

    def get_double_main_lines(self):
        """Get a set of lines forming a double line."""
        lines = list()
        xy = self.get_xy_line()
        style = deepcopy(self.style)

        # Make contour lines
        style['zorder'] = self.style.get('zorder', 0) -1
        style['linewidth'] = 1.8 * self.style.get('linewidth', 2)
        lines.append(mpl.lines.Line2D(*xy.transpose(), **style))

        # Make center lines
        style['zorder'] = self.style.get('zorder', 0)
        style['color'] = self.double_center_color
        style['linewidth'] = .5 * self.style.get('linewidth', 2)
        lines.append(mpl.lines.Line2D(*xy.transpose(), **style))
        return lines

    def get_single_main_lines(self):
        """Get the main lines."""
        xy = self.get_xy_line().transpose()
        line = mpl.lines.Line2D(*xy, **self.style)
        return [line]
        
    #def get_simple_main_lines(self):
    #    """Get the main lines."""
    #    xy = self.get_xy_line().transpose()
    #    line = mpl.lines.Line2D(*xy, **self.style)
    #    return [line]

    #def get_wiggly_main_lines(self):
    #    """Get the main lines."""
    #    xy = self.get_xy_line().transpose()
    #    line = mpl.lines.Line2D(*xy, **self.style)
    #    return [line]

    #def get_loopy_main_lines(self):
    #    """Get the main lines."""
    #    xy = self.get_xy_line().transpose()
    #    line = mpl.lines.Line2D(*xy, **self.style)
    #    return [line]

    def distance(self):
        """The distance between the starting point and the end point."""
        #return np.sqrt((self.ystart - self.yend) ** 2 + 
        #               (self.xstart - self.xend) ** 2)
        return np.linalg.norm(self.rend - self.rstart)

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

    def get_linepath(self, *args, **kwargs):
        """
        Get xy vectors for the path.

        Returns
        -------
        xy: np.ndarray of shape (N, 2)

"""
        if self.pathtype in ('linear', 'l', 'straight'):
            return self.get_linear_linepath(*args, **kwargs)
        elif self.pathtype in ('elliptic', 'e', 'ellipse'):
            return self.get_elliptic_linepath(*args, **kwargs)
        elif self.pathtype in ('circular', 'c', 'circle'):
            return self.get_circular_linepath(*args, **kwargs)
        else:
            raise ValueError('Wrong value for pathtype')

    def get_path_point(self, t):
        """
        Get xy vector for a particular position along the path.

        t: Distance parameter along the path. 0. <= t <= 1.

        Returns: np.ndarray of shape (2)
"""
        # This function is not optimized at all.
        linepath = self.get_linepath()
        i_xy = min(int(t * self.numpoints), self.numpoints)
        xy = linepath[i_xy]
        return xy.reshape(2)
        
    def get_linear_linepath(self):
        """Get xy vectors for the path."""
        dr = self.rend - self.rstart
        v, t = np.meshgrid(dr, self.t)
        path = self.rstart + v * t

        return path
        
    def get_elliptic_linepath(self):
        """Get xy vectors for the path."""

        # Geometry of the starting and end points
        dr = self.rend - self.rstart
        dx, dy = dr
        l = self.distance()

        # Line angle
        gamma = np.arctan(dy / dx)
        if dx < 0:
            gamma += np.pi

        R = np.array([[np.cos(gamma), - np.sin(gamma)],
                      [np.sin(gamma),   np.cos(gamma)]])

        # Ellipse center
        ro = self.rstart + dr / 2

        # Axes of the ellipse
        a = l / (2 * np.sin(self.alpha * np.pi))
        b = a / self.c * np.sign(self.d)

        # Angular progression along the ellipse.
        theta_s = np.pi * (1 - 2 * self.alpha) / 2.
        theta_i = 2 * np.pi * self.alpha
        theta = theta_s + theta_i * self.t

        # xy relative to the ellipse center
        ellipse = np.array([-a*np.cos(theta), b*np.sin(theta)]).transpose()

        # rotate ellipse and shift vector
        #path = np.dot(R, ellipse).transpose() + rot
        ellipse = vectors.sqdot(ellipse, R)
        path = vectors.add(ellipse, ro)

        return path

    def get_circular_linepath(self):
        """Get xy vectors for the path."""

        r = self.r

        # Circle center  # TODO: make use of fload value for d to tilt circle.
        ro = self.rend + np.array([0,1]) * self.d * self.r

        # Angular progression along the circle.
        theta_s = 0.
        theta_i = 2 * np.pi
        theta = theta_s + theta_i * self.t

        # xy relative to the circle center
        circle = r * np.array([- np.sin(theta), - np.cos(theta)]).transpose()

        # shift vector  # TODO add rotation of the circle (!)
        path = vectors.add(circle,  ro)

        return path

    def get_tangent(self, linepath=None, **kwargs):
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
        # TODO compute elliptic_tangent.
        return self.get_tangent_numeric(linepath, **kwargs)

    def get_tangent_point(self, t):
        """
        Get a particular tangent vector for a point along the path.

        t: Distance parameter along the path. 0. <= t <= 1.

        Returns: np.ndarray of shape (2)
"""
        tangent = self.get_tangent()
        i_v = min(int(t * self.numpoints), self.numpoints)
        v = tangent[i_v]
        return v.reshape(2)

    def get_tangent_numeric(self, linepath=None):
        """Compute tangent numerically."""
        if linepath is None:
            linepath = self.get_linepath()

        v = np.zeros((self.numpoints, 2))
        v[:-1] = linepath[1:] - linepath[:-1]
        v[-1] = linepath[-1] - linepath[-2]  # Gross approximation.

        # normalize the normal
        norm = np.sqrt(sum(v.transpose() * v.transpose()))
        tangent = vectors.dot(v, 1. / norm)

        return tangent

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
        if tangent is None:
            tangent = self.get_tangent()

        # Compute normal by rotation of the tangent.
        R = np.array([[0., -1.],[1., 0.]])
        normal = vectors.sqdot(tangent, R)

        return normal

    def get_normal_point(self, t):
        """
        Get a particular normal vector for a point along the path.

        t: Distance parameter along the path. 0. <= t <= 1.

        Returns: np.ndarray of shape (2)
"""
        normal = self.get_normal()
        i_v = min(int(t * self.numpoints), self.numpoints)
        v = normal[i_v]
        return v.reshape(2)

    def get_xy_line(self):
        """
        Get the xy vectors for the line points.

        Returns
        -------
            np.ndarray of shape (N, 2)

"""
        if self.linestyle == 'simple' or self.linestyle == 'double':
            return self.get_simple_xy_line()
        elif 'wiggly' in self.linestyle:
            return self.get_wiggly_xy_line()
        elif 'loopy' in self.linestyle:
            return self.get_loopy_xy_line()
        else:
            raise ValueError('Wrong value for linestyle.')

    def get_simple_xy_line(self):
        """Get the xy vectors for the line points."""
        return self.get_linepath()

    def get_wiggly_xy_line(self):
        """Get the xy vectors for the line points."""
        t = self.t
        linepath = self.get_linepath()
        tangent = self.get_tangent(linepath)
        normal = self.get_normal(tangent)

        # Number of waves
        numhalfwaves = int(2 * self.nwiggles)
        omega = np.pi * numhalfwaves
        phi = 2 * np.pi * self.phase

        sine = np.sin(omega * t + phi)

        dxy = self.amplitude * vectors.dot(normal, sine)
        dxy = vectors.add(dxy, -dxy[0])
        line = linepath + dxy

        return line

    def get_loopy_xy_line(self):
        """Get the xy vectors for the line points."""
        t = self.t
        linepath = self.get_linepath()
        tangent = self.get_tangent(linepath)
        normal = self.get_normal(tangent)

        # Number of waves
        omega = 2 * np.pi * self.nloops
        phi = 2 * np.pi * self.phase

        dy = - np.cos(omega * t + phi)
        dy -= dy[0]

        dx = np.sin(omega * t + phi)
        dx -= dx[0]

        dxy = (self.xamp * vectors.dot(tangent, dx) +
               self.yamp * vectors.dot(normal, dy))
        line = linepath + dxy
 
        return line

    def get_lines(self):
        """Get the lines."""
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

        self.c = kwargs.pop('c')

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
        height = width / self.c
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
    """A diagram. Can be a global object."""

    def __init__(self, fig=None, ax=None):

        if fig is None:
            self.fig = plt.gcf()

        if ax is None:
            self.ax = plt.gca()

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

