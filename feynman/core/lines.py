

from copy import copy, deepcopy

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

# Personnal modules
import mycolors as mc

from .. import vectors
from .. import colors
from ..constants import tau, pi

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
        The angle (in units of tau) spread by the ellipse arc.
        The limit cases are
            0.0 --> the curve will tend to a straight ligne,
            0.5 --> the curve will be half an ellipse,
            1.0 --> the curve will tend to a closed ellipse. 

    ellipse_excentricity : (1.2)
        The excentricity of the ellipse, that is, the ratio of the long axe
        over the short axe. At 1.0, the curve will be a circle arc.
        Also Controls wether the curve is 'up' or 'down'.
        A positive value makes it 'up', while a negative value makes it 'down'.

    ellipse_position :  ['up' | 1] or ['down', -1]
        In case ellipse_excentricity is not defined:

    circle_radius : float (.1)
        The radius of the circle.

    circle_angle : float (0.25)
        The angle of the anchor verticle to the circle center, in units of tau.

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
        Phase in the wiggly or loopy pattern, in units of tau.

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

    _pathtype_linear_aliases = ('linear','line',  'lin', 'l',)
    _pathtype_elliptic_aliases = ('elliptic', 'ellipse', 'ell', 'e')
    _pathtype_circular_aliases = ('circular', 'circle', 'cir', 'c')

    _linetype_simple_aliases = ('simple', 's', 'straight')
    _linetype_wiggly_aliases = ('wiggly', 'w', 'wiggle')
    _linetype_loopy_aliases = ('loopy', 'l', 'loop')

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
            arrow=True,
            npoints=400,
            ellipse_spread=.5,
            ellipse_excentricity=1.2,
            ellipse_position='up',
            # Conditional default : depends on: pathtype, linetype
            circle_radius=.15,
            circle_angle=0.25,
            amplitude=.025,
            xamp=.025,
            yamp=.05,
            nwiggles=5,
            nloops=14,
            phase=0,
            )

        # Adjust some default values according to pathtype and linetype
        if kwargs.get('linetype') in self._linetype_simple_aliases:
            default.update(arrow=True)
        elif (kwargs.get('linetype') in self._linetype_wiggly_aliases
           or kwargs.get('linetype') in self._linetype_loopy_aliases):
            default.update(arrow=False)

        if kwargs.get('pathtype') in self._pathtype_circular_aliases:
            if kwargs.get('linetype') == 'simple':
                default.update(circle_radius=.1)
            elif kwargs.get('linetype') == 'wiggly':
                default.update(nwiggles=7.25)
                default.update(phase=.75)
                default.update(circle_radius=.15)

        elif kwargs.get('pathtype') in self._pathtype_elliptic_aliases:
            if ('ellipse_position' in kwargs and
                kwargs['ellipse_position'] in ('down', -1)
                ):
                if 'ellipse_excentricity' not in kwargs:
                    default['ellipse_excentricity'] *= -1

        # Set default values
        for key, val in default.items():
            kwargs.setdefault(key, val)

        # Set attributes values
        for key in (

            # Options
            'arrow',

            # Path parameters
            'ellipse_spread',
            'ellipse_excentricity',
            'ellipse_position',
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
        self.arrows = list()

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

        # Add the arrow
        if self.arrow:
            _arrow_kwargs = """
                arrow_t
                arrow_width
                arrow_length
                arrow_fancyness
                """.split()

            arrow_kwargs = {}
            for key in _arrow_kwargs:
                skey = key.split('arrow_')[-1]
                if skey in kwargs:
                    arrow_kwargs[skey] = kwargs.pop(key)
            self.add_arrow(**arrow_kwargs)

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

    def t_index(self, t):
        return int(t * self.npoints)

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
        """The angle (units of tau) of the direct line between end points."""
        dx, dy = self.dr
        angle = np.arctan(dy / dx)
        if dx < 0: angle += np.pi
        return angle / (tau)

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
        R = vectors.rotation_matrix(.25)
        self._normal = vectors.sqdot(R, self.tangent)

    def set_pathtype(self, pathtype):
        """
        Set the path type.

            linear    -  A straight line between two points.
            elliptic  -  An ellipse arc.
            circular  -  A circle starting and ending at the same verticle.
"""
        if pathtype in      self._pathtype_linear_aliases:
            self._set_linepath = self._set_linear_linepath
        elif pathtype in    self._pathtype_elliptic_aliases:
            self._set_linepath = self._set_elliptic_linepath
        elif pathtype in    self._pathtype_circular_aliases:
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

    def add_arrow(self, t=0.5, direction=1, theta=.083, size=.025,
                  *args, **kwargs):
        """
        Add an arrow on the line.

        Arguments
        ---------

        arrowstyle : ['normal', 'fancy', 'line']

        t : float
            The position of the arrow along the line.
            Must be in the range [0,1].

        direction :
            The direction of the arrow. A positive number gives
            a forward arrow while a negative number gives a backward arrow.

        theta :
            The angle the arrow branches make with the path
            in units of tau

        size :
            The length of the arrow branches.

        width :

        length :
"""

        _arrow_style = 'normal', 'fancy', 'line'

        arrowstyle = kwargs.pop('arrowstyle', 'fancy')

        if arrowstyle == 'normal':
            return self._add_normal_arrow(*args, **kwargs)
        elif arrowstyle == 'fancy':
            return self._add_fancy_arrow(*args, **kwargs)
        elif arrowstyle == 'line':
            return self._add_line_arrow(*args, **kwargs)
        else:
            raise ValueError("Wrong value for arrowstyle. Allowed values : " + self._arrow_style)

    def _add_normal_arrow(self, t=.5, width=.03, length=.09, **kwargs):
        """
        Add a normal, triangular arrow.

        t :
        width :
        length :
        **kwargs :
"""

        for key, val in dict(
            color='k',
            zorder=12,
            ).items():
            kwargs.setdefault(key, val)

        center = self.path_point(t)
        tip = center + .5 * length * self.tangent_point(t)
        back = tip - .5 * length * self.tangent_point(t)
        c1 = back + .5 * width * self.normal_point(t)
        c2 = back - .5 * width * self.normal_point(t)

        arrow = mpa.Polygon([tip, c1, c2], **kwargs)

        self.arrows.append(arrow)

    def _add_fancy_arrow(self, t=.5, width=.03, length=.09, fancyness=.09, **kwargs):
        """
        Add a fancy arrow.

        Arguments
        ---------

        t :
        width :
        length :
        fancyness :
"""

        for key, val in dict(
            color='k',
            zorder=12,
            ).items():
            kwargs.setdefault(key, val)

        center = self.path_point(t)
        tangent = self.tangent_point(min(1, t+.025))
        normal = self.normal_point(t)

        tip = center + .5 * length * tangent
        back = tip - .5 * length * tangent
        c1 = back + .5 * width * normal
        c2 = back - .5 * width * normal
        antitip = back + fancyness * length * self.tangent_point(t)

        arrow = mpa.Polygon([tip, c1, antitip, c2], **kwargs)

        self.arrows.append(arrow)



    def _add_line_arrow(self, **kwargs):
        """
        Add an arrow made with lines.
"""
        raise NotImplementedError()


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

            th = tau * theta

            it = self.t_index(t)

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
        #i0 = int(self.npoints / 25)  # TODO use the line width for the offset
        #lines.append(mpl.lines.Line2D(x[i0:-i0], y[i0:-i0], **style))
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

    def path_point(self, t): return self.get_path_point(t)
    
        
    def _set_linear_linepath(self):
        """Compute the xy eectors for a linear line path."""
        v, t = np.meshgrid(self.dr, self.t)
        self.linepath = self.rstart + v * t
        
    def _set_elliptic_linepath(self):
        """Compute the xy vectors for an elliptic line path."""

        # Geometry of the starting and end points
        l = self.distance()

        # Rotation matrix.
        R = vectors.rotation_matrix(self.angle)

        # Ellipse center
        ro = self.rstart + self.dr / 2

        # Axes of the ellipse
        a = l / (2 * np.sin(self.ellipse_spread * np.pi))
        b = a / self.ellipse_excentricity

        # Angular progression along the ellipse.
        theta_s = np.pi * (1 - 2 * self.ellipse_spread) / 2.
        theta_i = tau * self.ellipse_spread
        theta = theta_s + theta_i * self.t

        # xy relative to the ellipse center
        ellipse = np.array([-a*np.cos(theta), b*np.sin(theta)]).transpose()

        # rotate ellipse and shift vector
        ellipse = vectors.sqdot(R, ellipse)
        self.linepath = vectors.add(ellipse, ro)

        return

    def _set_circular_linepath(self):
        """Get xy vectors for the path."""

        r = self.circle_radius
        alpha = self.circle_angle

        # Circle center
        ro = self.rend + vectors.rotate([r,0], alpha)

        # Angular progression along the circle.
        theta = tau * (self.t + alpha)

        # xy relative to the circle center
        circle = r * np.array([- np.cos(theta), - np.sin(theta)]).transpose()

        # shift vector
        self.linepath = vectors.add(circle,  ro)

        return

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

    def tangent_point(self, t): return self.get_tangent_point(t)

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

    def normal_point(self, t): return self.get_normal_point(t)

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
        if linetype in self._linetype_simple_aliases:
            self._set_xy = self._set_xy_simple
            return
        elif linetype in self._linetype_wiggly_aliases:
            self._set_xy = self._set_xy_wiggly
            return
        elif linetype in self._linetype_loopy_aliases:
            self._set_xy = self._set_xy_loopy
            return
        else:
            raise ValueError('Wrong value for linetype')

    def _set_xy_simple(self):
        """Compute the xy simple path.""" 
        self.xy = self.linepath

    def _set_xy_wiggly(self):
        """Compute the xy wiggly path.""" 

        numhalfwaves = int(2 * self.nwiggles)

        omega = np.pi * numhalfwaves
        phi = tau * self.phase

        sine = np.sin(omega * self.t + phi)

        dxy = self.amplitude * vectors.dot(self.normal, sine)
        dxy = vectors.add(dxy, -dxy[0])

        self.xy = self.linepath + dxy

    def _set_xy_loopy(self):
        """Compute the xy wiggly path.""" 

        omega = tau * self.nloops
        phi = tau * self.phase

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

    def get_arrows(self):
        """Get the patches, such as arrows."""
        return self.arrows

    def draw(self, ax):
        """Plot the line."""

        # Lines
        for line in self.get_lines():
            ax.add_line(line)

        # Arrows
        for arrow in self.get_arrows():
            ax.add_patch(arrow)

        # Text
        for text in self.get_texts():
            ax.add_artist(text)

        return

