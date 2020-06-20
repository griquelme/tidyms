from setuptools import setup
setup(name='ms_feature_validation',
      packages=['ms_feature_validation'],
      install_requires=['pandas>=1.0', 'pyopenms>=2.4', 'numpy>=1.15',
                        'pyyaml', 'statsmodels', 'scipy', 'scikit-learn>=0.23',
                        'bokeh', 'xlrd>=1.2', 'cerberus', 'seaborn',
                        'pytest>=5.0'])
