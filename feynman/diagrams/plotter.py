import matplotlib as mpl
import matplotlib.pyplot as plt

class Plotter(object):
    """
    A wrapper for matplotlib.figure.Figure object.
    as well as a styliser
    """

    def _init_figure(self, fig=None, ax=None, **kwargs):
        """Init internal figure object."""
        # Init diagram and ax

        if ax:
            self.ax = ax
            self.fig = ax.get_figure()
        else:
            self.fig = plt.figure()
            self.ax = self.fig.add_axes([0., 0., 1.,1.])
            for spine in self.ax.spines.values():
                spine.set_visible(False)
        
        self.set_ticks = kwargs.get('set_ticks', True)

        if self.set_ticks:
            self.ax.set_xticks([])
            self.ax.set_yticks([])

        self.transparent_background = kwargs.get('transparent', False)

    #def set_background_visible(self, transparent=True):
    #    """Set the background visibility."""
    #    self.transparent = transparent

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
    
    def singlesave(self, dname, ftype='pdf', show=False):
        """Save a single figure"""
        fname = self.get_fname(ftype, dname)
        self.check_mkdir(fname)
        self.fig.savefig(fname)
        if show:
            from subprocess import call
            call(['open', fname])
    
    def multisave(self, dname, ftype=('.svg', '.pdf'), show=True, showtype='pdf'):
        """Save figure under multiple formats."""
        for ftype in ('svg', 'pdf'):
            fname = self.get_fname(ftype, dname)
            self.singlesave(fname, show=(ftype == showtype))
