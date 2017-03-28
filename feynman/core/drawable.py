
__all__ = ['Drawable']

class Drawable(object):
    """
    A drawable object that belongs to a certain diagram.
    """

    _diagram = None

    @property
    def diagram(self):
        """The diagram it belongs to."""
        if not self._diagram:
            raise Exception('Diagram not found.')
        return self._diagram

    @diagram.setter
    def diagram(self, D):
        self._diagram = D

    def draw(self):
        pass
