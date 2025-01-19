import os
from setuptools import find_packages

install_requires = [
    "matplotlib>=1.1",
    "numpy>=1.6",
    ]

def find_package_data(dirname):
    paths = []
    for (path, directories, filenames) in os.walk(dirname):
        for filename in filenames:
            # We need the path relative to the main source directory.
            paths.append(os.path.join('..', path, filename))
    return paths

my_package_data = {'' :
    find_package_data('feynman/tests/baseline_images/'),
    }

setup_args = dict(
      name             = 'feynman',
      version          = '2.1.1',
      description      = 'Feynman diagrams with python-matplotlib.',
      author           = 'Gabriel Antonius',
      author_email     = 'gabriel.antonius@gmail.com',
      license          = 'GPL',
      keywords         = 'Feynman diagrams',
      url              = 'http://gkantonius.github.io/feynman',
      install_requires = install_requires,
      packages         = find_packages(),
      package_data     = my_package_data,
      )


if __name__ == "__main__":
    from setuptools import setup
    setup(**setup_args)

