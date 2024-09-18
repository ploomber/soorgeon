import re
import ast
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup

_version_re = re.compile(r"__version__\s+=\s+(.*)")

with open("src/soorgeon/__init__.py", "rb") as f:
    VERSION = str(
        ast.literal_eval(_version_re.search(f.read().decode("utf-8")).group(1))
    )

REQUIRES = [
    "ploomber-core>=0.0.4",
    "jupytext",
    "parso",
    "nbformat",
    "jinja2",
    "pyyaml",
    "click",
    "isort",
    # for checking code errors
    "pyflakes",
    "black[jupyter]>=22.6.0",
    "papermill",
]

DEV = [
    "pkgmt",
    "pytest",
    "yapf",
    "flake8",
    "invoke",
    "twine",
    "ipython",
    "ploomber",
    # to download data for running _kaggle notebooks
    "kaggle",
    # to fetch from github repo
    "pygithub",
    # to run some of the examples
    "pandas",
    "scikit-learn",
    "seaborn",
    "pkgmt",
    "twine",
]

DESCRIPTION = "Convert monolithic Jupyter notebooks" " into maintainable pipelines."

setup(
    name="soorgeon",
    version=VERSION,
    description=DESCRIPTION,
    license=None,
    author="Eduardo Blancas",
    author_email="hello@ploomber.io",
    url="https://github.com/ploomber/soorgeon",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    package_data={"": []},
    classifiers=[],
    keywords=[],
    install_requires=REQUIRES,
    extras_require={
        "dev": DEV,
    },
    entry_points={
        "console_scripts": ["soorgeon=soorgeon.cli:cli"],
    },
)
