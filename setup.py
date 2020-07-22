from setuptools import setup
setup(name='tidyms',
      packages=['tidyms'],
      install_requires=['pandas>=1.0', 'pyopenms>=2.4', 'numpy>=1.15',
                        'pyyaml', 'statsmodels', 'scipy=1.4', 'scikit-learn>=0.23',
                        'bokeh', 'xlrd>=1.2', 'cerberus', 'seaborn',
                        'pytest>=5.0'])
