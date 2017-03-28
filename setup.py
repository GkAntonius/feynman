from setuptools import find_packages

install_requires = [
    "matplotlib>=1.1",
    "numpy>=1.6",
    ]

setup_args = dict(
      name             = 'feynman',
      version          = '1.3.0',
      description      = "Feynman diagrams with python-matplotlib.",
      author           = "Gabriel Antonius",
      author_email     = "gabriel.antonius@gmail.com",
     #url              = url,
     #license          = license,
     #platforms        = platforms,
     #keywords         = keywords,
      install_requires = install_requires,
      packages         = find_packages(),
      )


if __name__ == "__main__":
    from setuptools import setup
    setup(**setup_args)

