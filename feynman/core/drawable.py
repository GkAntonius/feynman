
__all__ = ['Drawable']

class Drawable(object):
    """
    A drawable object that belongs to a certain diagram.
    """

    def __init__(self):
        self._diagram = None

    @property
    def diagram(self):
        """The diagram it belongs to."""
        return self._diagram

    def draw(self):
        pass
