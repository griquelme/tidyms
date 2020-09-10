PACKAGE_NAME = "tidyms"
VERSION = "0.1.0"
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

PACKAGES = ["tidyms"]

PYTHON_REQUIRES = ">=3.6"

INSTALL_REQUIRES = [
    'pandas>=1.0',
    'pyopenms>=2.4',
    'numpy>=1.15',
    'pyyaml',
    'statsmodels',
    'scipy<=1.4.1',
    'scikit-learn>=0.23',
    'bokeh',
    'xlrd>=1.2',
    'cerberus',
    'seaborn',
    'pytest>=5.0'
    'openpyxl>=3.0'
]

if __name__ == "__main__":
    from setuptools import setup
    from sys import version_info

    if version_info[:2] < (3, 6):
        msg = "tidyms requires Python >= 3.6."
        raise RuntimeError(msg)

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
          packages=PACKAGES,
          python_requires=PYTHON_REQUIRES,
          install_requires=INSTALL_REQUIRES,
          include_package_data=True)
