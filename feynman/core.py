#
#
# TODO
# ----
#
#   o Verticle
#       - Style
#       - get_lines
#   o Line
#       - double {simple, wiggly, loopy} lines
#
#   o Operators
#   o Text
#   o 
#   o Scalability
#


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


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
    def __init__(self, xy=(0,0), *args, **kwargs):

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

    def draw(self, ax):
        ax.plot(*self.xy, **self.style)


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

    arrow: bool = True
        Include an arrow in the line.

    nwiggles : float
        The number of wiggles in a wiggly line.
        Can be integer or half-integer (if the phase is 0 or .5).

    nloops : float
        The number of loops in a loopy line.

    phase : float
        Phase in the wiggly or loopy pattern, in units of 2pi.


"""
    def __init__(self, vstart, vend, *args, **kwargs):

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
            kwarge.get('linestyle') == 'wiggly'):
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

        # 'matplotlib' line style
        self.style = dict(
            marker='',
            color='k',
            linestyle='-',
            linewidth=3,
            zorder=10,
            )
        self.style.update(kwargs)


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

        if self.arrow:
            self.add_arrow()

    @property
    def rstart(self):
        """The starting position."""
        return np.array(self.vstart.xy)

    @property
    def rend(self):
        """The end position."""
        return np.array(self.vend.xy)

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

        **args :
            Any style specification, such as linewidth.

"""
        if not (t >= 0 and t <= 1):
            raise ValueError("t should be in range [0,1]")
        param = (t, direction, theta, size, kwargs)
        self.arrows_param.append(param)

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

            xy = np.array([rstart, rtip, rend]).transpose()
            arrow_line = mpl.lines.Line2D(*xy, **style)

            lines.append(arrow_line)

        return lines


    def get_main_lines(self):
        """Get the main lines."""
        if self.linestyle in ('simple', 's'):
            return self.get_simple_main_lines()
        elif self.linestyle in ('wiggly', 'w'):
            return self.get_wiggly_main_lines()
        elif self.linestyle in ('loopy', 'l'):
            return self.get_loopy_main_lines()
        else:
            raise ValueError('Wrong value for linestyle.')
        
    def get_simple_main_lines(self):
        """Get the main lines."""
        xy = self.get_xy_line().transpose()
        style = self.style
        line = mpl.lines.Line2D(*xy, **self.style)
        return [line]

    def get_wiggly_main_lines(self):
        """Get the main lines."""
        xy = self.get_xy_line().transpose()
        style = self.style
        line = mpl.lines.Line2D(*xy, **self.style)
        return [line]

    def get_loopy_main_lines(self):
        """Get the main lines."""
        xy = self.get_xy_line().transpose()
        style = self.style
        line = mpl.lines.Line2D(*xy, **self.style)
        return [line]

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
        rot = np.tile(ro, (self.numpoints, 1))

        # Axes of the ellipse
        a = l / (2 * np.sin(self.alpha * np.pi))
        b = a / self.c * np.sign(self.d)

        # Angular progression along the ellipse.
        theta_s = np.pi * (1 - 2 * self.alpha) / 2.
        theta_i = 2 * np.pi * self.alpha
        theta = theta_s + theta_i * self.t

        # xy relative to the ellipse center
        ellipse = np.array([- a * np.cos(theta), b * np.sin(theta)])

        # rotate ellipse and shift vector
        path = np.dot(R, ellipse).transpose() + rot

        return path

    def get_circular_linepath(self):
        """Get xy vectors for the path."""

        r = self.r

        # Circle center  # TODO: make use of fload value for d to tilt circle.
        ro = self.rend + np.array([0,1]) * self.d * self.r
        rot = np.tile(ro, (self.numpoints, 1))

        # Angular progression along the circle.
        theta_s = 0.
        theta_i = 2 * np.pi
        theta = theta_s + theta_i * self.t

        # xy relative to the circle center
        circle = r * np.array([- np.sin(theta), - np.cos(theta)])

        # shift vector  # TODO add rotation of the circle (!)
        path = circle.transpose() + rot

        return path

    def get_tangent(self, linepath=None, *args, **kwargs):
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
        return self.get_tangent_numeric(*args, **kwargs)

    def get_tangent_numeric(self, linepath=None):
        """Compute tangent numerically."""
        if linepath is None:
            linepath = self.get_linepath()

        v = np.zeros((self.numpoints, 2))
        v[:-1] = linepath[1:] - linepath[:-1]
        v[-1] = linepath[-1] - linepath[-2]  # Gross approximation.

        # normalize the normal
        norm = np.sqrt(sum(v.transpose() * v.transpose()))
        normt = np.tile(norm, (2,1)).transpose()
        tangent = v / normt

        return tangent

    def get_normal(self, tangent=None, *args, **kwargs):
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
        normal = np.zeros((self.numpoints, 2))
        if tangent is None:
            tangent = self.get_tangent()

        # Compute normal by rotation of the tangent.
        R = np.array([[0., -1.],[1., 0.]])
        normal = np.dot(R, tangent.transpose()).transpose()

        return normal

    def get_xy_line(self):
        """
        Get the xy vectors for the line points.

        Returns
        -------
            np.ndarray of shape (N, 2)

"""
        if self.linestyle in ('simple', 's'):
            return self.get_simple_xy_line()
        elif self.linestyle in ('wiggly', 'w'):
            return self.get_wiggly_xy_line()
        elif self.linestyle in ('loopy', 'l'):
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
        sinet = np.tile(sine, (2, 1)).transpose()

        dxy = self.amplitude * sinet * normal
        shiftt = np.tile(dxy[0], (self.numpoints, 1))
        dxy -= shiftt
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
        dyt = np.tile(dy, (2, 1)).transpose()

        dx = np.sin(omega * t + phi)
        dx -= dx[0]
        dxt = np.tile(dx, (2, 1)).transpose()

        dxy = self.xamp * dxt * tangent + self.yamp * dyt * normal
        line = linepath + dxy

        return line



class Operator(object):
    """
    A N-point operator.
    Often represented as a polygon, or a circle.

"""
    def __init__(self):
        self.dimensions = 2
        self.shape = 'oval'
        self.shape = 'polygon'

    def draw(self, ax):
        """Draw the diagram."""


class Diagram(object):
    """A diagram. Can be a global object."""

    def __init__(self, fig=None, ax=None):

        self.boxposition = (0,0,1,1)

        if fig is None:
            self.fig = plt.gcf()

        if ax is None:
            self.ax = plt.gca()

        self.verticles = list()
        self.lines = list()
        self.operators = list()

    def verticle(self, *args, **kwargs):
        """Add a new verticle."""
        v = Verticle(*args, **kwargs)
        self.verticles.append(v)
        return v

    def line(self, v1, v2, *args, **kwargs):
        """Add a line."""
        l = Line(v1, v2, *args, **kwargs)
        self.lines.append(l)
        return l

    def plot(self):
        """Draw the diagram."""

        for v in self.verticles:
            v.draw(self.ax)

        for l in self.lines:
            l.draw(self.ax)

