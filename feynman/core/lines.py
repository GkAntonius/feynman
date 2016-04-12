
from copy import copy, deepcopy
import warnings

import numpy as np
from numpy import sin, cos, sqrt
from numpy.linalg import norm
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import matplotlib.text as mpt

from .. import vectors
from .. import colors
from ..constants import tau, pi
from .util import matplotlib_Line2D_valid_keyword_arguments


class FancyLine(object):
    """
    A particle line joigning two verticles.

    Arguments
    =========

    vstart :
        Starting vericle.

    vend :
        End vericle.


    style : "shape flavour stroke"
    ---------------------------------- 

    shape : ( linear)
        The shape determines the line path.

        |   linear    -  A straight line between two points.
        |   elliptic  -  An ellipse arc.
        |   circular  -  A circle starting and ending at the same verticle.

    flavour : ( simple )
        The type of line.

        |   simple  -  A straight line that follows the line path.
        |   wiggly  -  A wavy line around the line path..
        |   loopy   -  A spring.

    stroke : ( single )
        The style for the line.

        |   single  -  A simple line.
        |   double  -  A double line.


    ellipse_param  : "spread excentricity position"
    -----------------------------------------------

    ellipse_spread : (0.5)
        The angle (in units of tau) spread by the ellipse arc.
        The limit cases are
        |   0.0 --> the curve will tend to a straight ligne,
        |   0.5 --> the curve will be half an ellipse,
        |   1.0 --> the curve will tend to a closed ellipse. 

    ellipse_excentricity : (1.2)
        The excentricity of the ellipse, that is, the ratio of the long axe
        over the short axe. At 1.0, the curve will be a circle arc.
        Also Controls wether the curve is 'up' or 'down'.
        A positive value makes it 'up', while a negative value makes it 'down'.

    ellipse_position :  ['up' | 1] or ['down', -1]
        In case ellipse_excentricity is not defined:


    circle_param  : dict( radius=.1, angle=.25) 
    -----------------------------------------------

    circle_radius : float (.1)
        The radius of the circle.

    circle_angle : float (0.25)
        The angle of the anchor verticle to the circle center, in units of tau.

    circle_excentricity : float (.1)
        The excentricity of the circle.
        | > 1. --> squeezed in the tangent direction at the anchor point.
        | < 1. --> stretched in the tangent direction at the anchor point.


    arrow : bool ( True )
        Include an arrow in the line.

    arrow_param : dict
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

    **line_kwargs : 
        Keyword arguments for matplotlib.Lines.line2D .


    Properties
    ==========

    npoints : int
        Number of points that makes up the line.

    xy : np.ndarray, shape (npoints, 2)
        The xy coordinates of the line.

    linepath : np.ndarray, shape (npoints, 2)
        The director line arount which the xy points are placed (with line_kwargs).

    tangent : np.ndarray, shape (npoints, 2)
        The unitary vectors tangent to the linepath.

    normal : np.ndarray, shape (npoints, 2)
        The unitary vectors normal to the linepath.

    """

    _shape = 'linear'
    _shape_linear_aliases = ('straight', 'linear','line','l',)
    _shape_elliptic_aliases = ('elliptic', 'ellipse', 'ell', 'e')
    _shape_circular_aliases = ('circular', 'circle', 'cir', 'c')
    _shape_possible_values = sum([],_shape_linear_aliases + _shape_elliptic_aliases + _shape_circular_aliases)

    _flavour = 'simple'
    _flavour_simple_aliases = ('simple',) #, 'straight'
    _flavour_wiggly_aliases = ('wiggly', 'w', 'wiggle', 'wiggles')
    _flavour_loopy_aliases = ('loopy', 'l', 'loop', 'loops')
    _flavour_possible_values = sum([],_flavour_simple_aliases + _flavour_wiggly_aliases + _flavour_loopy_aliases)

    _stroke = 'single'
    _stroke_single_aliases = ('single', 's')
    _stroke_double_aliases = ('double', 'd')
    _stroke_possible_values = sum([], _stroke_single_aliases + _stroke_double_aliases)

    _xy = np.zeros((2, 2))
    _linepath = np.zeros((2, 2))
    _tangent = np.zeros((2,2))
    _normal = np.zeros((2,2))
    _main_lines = None

    t =  np.linspace(0, 1, 2)

    def __init__(self, vstart, vend, **kwargs):

        # Tips of the line
        self.vstart = vstart
        self.vend = vend

        # Set default values
        self.set_default()
        self.set_style(kwargs)
        self.set_shape_dependent_defaults(kwargs)
        self.set_kwargs_defaults(kwargs)

        self.set_attributes(kwargs)

        #  Compute various vectors
        self.set_t_param(kwargs)  # Set the main parameter
        self._set_linepath()  # Compute the linepath
        self._set_tangent()  # Compute the tangent
        self._set_normal()  # Compute the normal
        self._set_xy()  # Compute the line points

        # Initialize objects
        self.lines = list()
        self.patches = list()
        self.texts = list()
        self.arrows = list()

        # If requested
        self.add_initial_arrow(kwargs)

        # All other kwargs are 'matplotlib' line line_kwargs arguments
        self.set_line_kwargs_defaults(kwargs)

        # Collect unknown keyword arguments
        self.warn_unkwnown_kwargs(kwargs)

    def draw(self, ax):
        """Plot the line."""
        for line in self.get_lines(): ax.add_line(line)      # Lines
        for arrow in self.get_arrows(): ax.add_patch(arrow)  # Arrows
        for text in self.get_texts(): ax.add_artist(text)    # Texts
        return

    def set_style(self, kwargs):
        """Set at once the shape, flavour and stroke of the line."""

        def popdefault(D, key):
            default = self.default.pop(key)
            return D.pop(key, default)
        shape = popdefault(kwargs, 'shape')
        flavour = popdefault(kwargs, 'flavour')
        stroke = popdefault(kwargs, 'stroke')
        style = popdefault(kwargs, 'style')

        # Attempt to know shape
        for token in style.split():
            if token in self._shape_possible_values:
                shape = token
            elif token in self._flavour_possible_values:
                flavour = token
            elif token in self._stroke_possible_values:
                stroke = token

        self.set_shape(shape)
        self.set_flavour(flavour)
        self.set_stroke(stroke)
                
    def set_default(self, **overwrite):
        """Set the default values."""
        default = dict(
            stroke='single',
            flavour='simple',
            shape='linear',
            style='',
            arrow=True,
            npoints=400,
            ellipse_spread=.5,
            ellipse_excentricity=1.2,
            ellipse_position='up',
            # Conditional default : depends on: shape, flavour
            circle_radius=.15,
            circle_angle=0.25,
            circle_excentricity=1.,
            amplitude=.025,
            xamp=.025,
            yamp=.05,
            nwiggles=5,
            nloops=14,
            phase=0,
            double_center_color='w',
            )

        default.update(**overwrite)
        self.default = default

    def set_shape_dependent_defaults(self, kwargs):
        """Adjust some default values according to shape and flavour."""

        if self.flavour == 'simple':
            self.default.update(arrow=True)

        elif self.flavour in ('wiggly', 'loopy'):
            self.default.update(arrow=False)

        if self.shape == 'circular':
            if self.flavour == 'simple':
                self.default.update(circle_radius=.1)
            elif self.flavour == 'wiggly':
                self.default.update(nwiggles=7.25)
                self.default.update(phase=.75)
                self.default.update(circle_radius=.15)
            elif self.flavour == 'loopy':
                pass

        elif self.shape == 'elliptic':
            if ('ellipse_position' in kwargs and
                kwargs['ellipse_position'] in ('down', -1)
                ):
                if 'ellipse_excentricity' not in kwargs:
                    self.default['ellipse_excentricity'] *= -1

    def set_line_kwargs_defaults(self, kwargs):
        """Set default matplotlib.lines.Line2D keyword arguments."""
        self.line_kwargs = dict(
            marker='',
            color='k',
            linestyle='-',
            linewidth=3,
            zorder=10,
            solid_capstyle="butt",
            )

        for key in kwargs.keys():
            if key in matplotlib_Line2D_valid_keyword_arguments:
                self.line_kwargs.update({key : kwargs.pop(key)})

    def warn_unkwnown_kwargs(self, kwargs):
        """Collect unknown keyword arguments."""
        for key in kwargs.keys():
            warnings.warn("Unknown keyword argument will be ignored: " + str(key))

    def set_kwargs_defaults(self, kwargs):
        """Set default values into kwargs."""
        for key, val in self.default.items():
            kwargs.setdefault(key, val)

    def set_attributes(self, kwargs):
        """Set attributes values from keyword arguments."""
        for key in (

            # Options
            'arrow',

            # Path parameters
            'ellipse_spread',
            'ellipse_excentricity',
            'ellipse_position',
            'circle_radius',
            'circle_excentricity',
            'circle_angle',

            # Amplitude of the wave and such...
            'amplitude',
            'xamp',
            'yamp',

            # Wiggly and loopy line parameters
            'nwiggles',
            'nloops',
            'phase',

            'double_center_color',

            ):
            self.__dict__[key] = kwargs.pop(key)

    def add_initial_arrow(self, kwargs):
        """Add the arrow, if requested at initiation."""
        arrow_param = kwargs.pop('arrow_param', dict())
        if self.arrow:
            self.add_arrow(**arrow_param)


    @property
    def xstart(self): return self.vstart.xy[0]
    @property
    def ystart(self): return self.vstart.xy[1]
    @property
    def xend(self): return self.vend.xy[0]
    @property
    def yend(self): return self.vend.xy[1]


    def t_index(self, t):
        """Querry the index of a given value of t."""
        return int(t * self.npoints)

    def set_t_param(self, kwargs, **overwrite):
        kwargs.update(**overwrite)
        self._set_t(npoints=kwargs.pop('npoints', 2))

    def _set_t(self, npoints=2):
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

    @property
    def shape(self):
        return self._shape

    def set_shape(self, shape):
        """
        Set the path type.

            linear    -  A straight line between two points.
            elliptic  -  An ellipse arc.
            circular  -  A circle starting and ending at the same verticle.
        """
        if shape in self._shape_linear_aliases:
            self._set_linepath = self._set_linear_linepath
            self._shape = 'linear'
        elif shape in self._shape_elliptic_aliases:
            self._set_linepath = self._set_elliptic_linepath
            self._shape = 'elliptic'
        elif shape in self._shape_circular_aliases:
            self._shape = 'circular'
            self._set_linepath = self._set_circular_linepath
        else:
            raise ValueError('Wrong value for shape')

    # User
    def get_xy(self):
        """Return the xy array of the line points, of shape (numpoins, 2)."""
        return self.xy

    # User
    def set_xy(self, xy):
        """Set the xy array of the line points, of shape (numpoins, 2)."""
        self.xy = xy


    @property
    def flavour(self):
        return self._flavour

    def set_flavour(self, flavour):
        """
        Set the line style.

            simple  -  A straight line.
            wiggly  -  A wavy line.
            loopy   -  A spring.
        """
        if flavour in self._flavour_simple_aliases:
            self._set_xy = self._set_xy_simple
            self._flavour = 'simple'
            return
        elif flavour in self._flavour_wiggly_aliases:
            self._set_xy = self._set_xy_wiggly
            self._flavour = 'wiggly'
            return
        elif flavour in self._flavour_loopy_aliases:
            self._set_xy = self._set_xy_loopy
            self._flavour = 'loopy'
            return
        else:
            raise ValueError('Wrong value for flavour')

    #   -------------------------------------------------------------------   #
    #   ------------------ Arrows routines --------------------------------   #
    #   -------------------------------------------------------------------   #

    def add_arrow(self, #t=0.5, direction=1, theta=.083, size=.025,
                  *args, **kwargs):
        """
        Add an arrow on the line.

        Arguments
        ---------

        style : ['normal', 'fancy', 'line']

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

        _style = 'normal', 'fancy', 'line'

        style = kwargs.pop('style', 'fancy')

        if style == 'normal':
            return self._add_normal_arrow(*args, **kwargs)
        elif style == 'fancy':
            return self._add_fancy_arrow(*args, **kwargs)
        elif style == 'line':
            return self._add_line_arrow(*args, **kwargs)
        else:
            raise ValueError("Wrong value for style.\n Allowed values : " + self._style)

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

    def _add_fancy_arrow(self, t=.5, width=.03, length=.09, fancyness=.09,
                         t_shift_dir=0.025, **kwargs):
        """
        Add a fancy arrow.

        Arguments
        ---------

        t : Where to place the arrow along the line. [0, 1]
        t_shift_dir : 
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
        tangent = self.tangent_point(min(1, t+t_shift_dir))
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
        kwargs.setdefault('fontsize', 14)
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

    @property
    def main_lines(self):
        return self._main_lines

    @property
    def stroke(self):
        return self._stroke

    def set_stroke(self, stroke):
        """
        Set the stroke.

            single  -  A single line.
            double  -  A double line.
        """
        if stroke in self._stroke_single_aliases:
            self.get_main_lines = self.get_single_main_lines
            self._stroke = 'single'
        elif stroke in self._stroke_double_aliases:
            self.get_main_lines = self.get_double_main_lines
            self._stroke = 'double'
        else:
            raise ValueError('Wrong value for stroke: ' + str(stroke))

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
        style = deepcopy(self.line_kwargs)
        x, y = self.xy.transpose()

        # Make contour lines
        style['zorder'] = self.line_kwargs.get('zorder', 0) -1
        style['linewidth'] = 1.8 * self.line_kwargs.get('linewidth', 2)
        lines.append(mpl.lines.Line2D(x, y, **style))

        # Make center lines
        style['zorder'] = self.line_kwargs.get('zorder', 0)
        style['color'] = self.double_center_color
        style['linewidth'] = .5 * self.line_kwargs.get('linewidth', 2)
        lines.append(mpl.lines.Line2D(x, y, **style))
        return lines

    def _set_double_main_lines(self):
        self.main_lines = get_double_main_lines()

    def get_single_main_lines(self):
        """Get the main lines."""
        style = dict()
        for key, val in self.line_kwargs.items():
            if key in matplotlib_Line2D_valid_keyword_arguments:
                style[key] = val

        line = mpl.lines.Line2D(*self.xy.transpose(), **style)
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
        norms = sqrt(sum(dl.transpose() * dl.transpose()))
        l = sum(norms)
        return l

    def linelength(self, linepath=None):
        """The length of the path."""
        if line is None:
            line = self.get_line()

        # Integrate the line
        dl = line[1:] - line[:-1]
        norms = sqrt(sum(dl.transpose() * dl.transpose()))
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
    
    #   -------------------------------------------------------------------   #
    #   ------------------ linepath routines ------------------------------   #
    #   -------------------------------------------------------------------   #
        
    def _set_linear_linepath(self):
        """Compute the xy eectors for a linear line path."""
        v, t = np.meshgrid(self.dr, self.t)
        self.linepath = self.rstart + v * t
        
    def _set_elliptic_linepath(self):
        """Compute the xy vectors for an elliptic line path."""

        # Geometry of the starting and end points
        l = self.distance()
        c = self.ellipse_excentricity

        # Rotation matrix.
        R = vectors.rotation_matrix(self.angle)

        # Axes of the ellipse
        alpha = self.ellipse_spread * tau / 2
        beta = tau / 4 - alpha

        a = l / (2 * cos(beta))
        b = a / c

        # Angular progression along the ellipse.
        theta = tau/2 - beta - 2 * alpha * self.t

        # xy relative to the ellipse center
        ellipse = np.array([a * cos(theta), b * sin(theta)]).transpose()

        # Ellipse center
        ro = (self.rstart + self.dr / 2 +
              vectors.rotate(self.dr, -.25) * b * sin(beta) / norm(self.dr))

        # rotate ellipse and shift vector
        ellipse = vectors.sqdot(R, ellipse)
        ellipse = vectors.add(ellipse, ro)

        self.linepath = ellipse

        return

    def _set_circular_linepath(self):
        """Get xy vectors for the path."""

        r = self.circle_radius
        c = self.circle_excentricity
        alpha = self.circle_angle

        # Circle center
        ro = self.rend + vectors.rotate([r,0], alpha)

        # Angular progression along the circle.
        #theta = tau * (self.t + alpha)
        theta = tau * self.t

        # xy relative to the circle center
        circle = r * np.array([- cos(theta), - sin(theta) / c])#.transpose()
        circle = vectors.rotate(circle, alpha)
        circle = circle.transpose()

        # shift vector
        self.linepath = vectors.add(circle,  ro)

        return


    #   -------------------------------------------------------------------   #
    #   ------------------ User routines ----------------------------------   #
    #   -------------------------------------------------------------------   #
        
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


    #   -------------------------------------------------------------------   #
    #   -------------------- core routines ----------------------------------   #
    #   -------------------------------------------------------------------   #
        
    # Morphable
    def _set_xy(self):
        """Compute the xy coordinates of the line."""
        return self._set_xy_simple()

    def _set_xy_simple(self):
        """Compute the xy simple path.""" 
        self.xy = self.linepath

    def _set_xy_wiggly(self):
        """Compute the xy wiggly path.""" 

        numhalfwaves = int(2 * self.nwiggles)

        omega = np.pi * numhalfwaves
        phi = tau * self.phase

        sine = sin(omega * self.t + phi)

        dxy = self.amplitude * vectors.dot(self.normal, sine)
        dxy = vectors.add(dxy, -dxy[0])

        self.xy = self.linepath + dxy

    def _set_xy_loopy(self):
        """Compute the xy wiggly path.""" 

        omega = tau * self.nloops
        phi = tau * self.phase

        dy = - cos(omega * self.t + phi)
        dy -= dy[0]

        dx = sin(omega * self.t + phi)
        dx -= dx[0]

        dxy = (self.xamp * vectors.dot(self.tangent, dx) +
               self.yamp * vectors.dot(self.normal, dy))

        self.xy = self.linepath + dxy

    def get_lines(self):
        """Get the lines."""  # Could be faster
        lines = list()
        lines.extend(self.get_main_lines())
        #lines.extend(self.get_arrow_lines())
        return lines

    def get_arrows(self):
        """Get the patches, such as arrows."""
        return self.arrows

