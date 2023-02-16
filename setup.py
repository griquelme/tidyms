PACKAGE_NAME = "tidyms"
VERSION = "0.6.2"
LICENSE = 'BSD (3-clause)'
AUTHOR = "Bioanalytical Mass Spectrometry Group at CIBION-CONICET"
AUTHOR_EMAIL = "griquelme.chm@gmail.com"
MAINTAINER = "Gabriel Riquelme"
MAINTAINER_EMAIL = AUTHOR_EMAIL
URL = "https://github.com/griquelme/tidyms"
DESCRIPTION = "Tools for working with MS data in metabolomics"

with open("README.md") as fin:
    LONG_DESCRIPTION = fin.read()
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

CLASSIFIERS = [
    "License :: OSI Approved :: BSD License",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "Topic :: Scientific/Engineering :: Medical Science Apps."
]

PYTHON_REQUIRES = ">=3.9"

INSTALL_REQUIRES = [
    "bokeh>=2.4",
    "Cerberus>=1.3",
    "ipython>=8.1",
    "joblib>=1.1",
    "matplotlib>=3.5.1",
    "networkx>=3.0",
    "numpy>=1.22",
    "openpyxl>=3.0",
    "pandas>=1.4.1",
    "requests",
    "scikit-learn>=1.0.2",
    "scipy>=1.8",
    "seaborn>=0.11",
    "statsmodels>=0.13",
    "tqdm>=4.0",
    "xlrd>=2.0"
]

if __name__ == "__main__":
    from setuptools import setup, find_packages
    from sys import version_info
    # from Cython.Build import cythonize

    if version_info[:2] < (3, 9):
        msg = "tidyms requires Python >= 3.9."
        raise RuntimeError(msg)

    # ext_modules = cythonize(["tidyms/chem/*.pyx"],
    #                         compiler_directives={'language_level': "3"})

    setup(name=PACKAGE_NAME,
          version=VERSION,
          author=AUTHOR,
          author_email=AUTHOR_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          license=LICENSE,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
          url=URL,
          classifiers=CLASSIFIERS,
          packages=find_packages(),
          python_requires=PYTHON_REQUIRES,
          install_requires=INSTALL_REQUIRES,
          include_package_data=True,
<<<<<<< HEAD
          
          setup_requires=["pytest-runner"],
          tests_require=["pytest"],)
=======
          # ext_modules=ext_modules
    )
>>>>>>> master
