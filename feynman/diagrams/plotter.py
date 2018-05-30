import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy import array

class Plotter(object):
    """
    A wrapper for matplotlib.figure.Figure object.
    """

    def _init_figure(self, ax=None, **kwargs):
        """Init internal figure object."""

        # Init diagram and ax
        if ax:
            self.ax = ax
            self.fig = ax.get_figure()
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_axes([0., 0., 1.,1.])
            self.set_size_inches(*kwargs.get('figsize', (6, 6)))
            self.ax.set_xlim(0,1)
            self.ax.set_ylim(0,1)
            for spine in self.ax.spines.values():
                spine.set_visible(False)
        
        self.set_ticks = kwargs.get('set_ticks', True)

        if self.set_ticks:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        self.transparent_background = kwargs.get('transparent', False)

        self.x0 = 0.
        self.y0 = 0.

    def show(self):
        """Show the figure with matplotlib.pyplot.show."""
        plt.show()

    def gcf(self):
        """Get the figure."""
        return self.fig

    def gca(self):
        """Get the axe."""
        return self.ax

    @staticmethod
    def get_fname(ftype, diagname, where='.'):
        """Return a file name for a diagram."""
        from os.path import join as pjoin
        return pjoin('{}'.format(where), '{}.{}'.format(diagname, ftype))
    
    @staticmethod
    def check_mkdir(fname):
        """Op """
        from os.path import exists, dirname
        dirnm = dirname(fname)
        if not exists(dirnm):
            from subprocess import call
            call(['mkdir', '-p', dirnm])

    def savefig(self, fname, **kwargs):
        return self.fig.savefig(fname, **kwargs)
    
    def singlesave(self, dname, ftype='pdf', show=False):
        """Save a single figure"""
        fname = self.get_fname(ftype, dname)
        self.check_mkdir(fname)
        self.fig.savefig(fname, transparent=self.transparent_background)
        if show:
            from subprocess import call
            call(['open', fname])
    
    def multisave(self, dname, ftype=('.svg', '.pdf'), show=True, showtype='pdf'):
        """Save figure under multiple formats."""
        for ftype in ('svg', 'pdf'):
            self.singlesave(dname, ftype, show=(ftype == showtype))

    def set_size_inches(self, w=8, h=6):
        """Set the figure size, and set xlim, ylim, x0 and y0 accordingly."""
        # Geometry
        aspectratio = float(h) / float(w)
        self.fig.set_size_inches(w, h)

        self.ax.set_xlim(.0, w)

        #self.ax.set_xlim(.0, 10.)
        self.ax.set_ylim(array(self.ax.get_xlim()) * aspectratio)
        self.y0 = sum(self.ax.get_ylim()) / 2.
        self.x0 = sum(self.ax.get_xlim()) * .05

