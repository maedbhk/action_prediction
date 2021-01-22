social_eye
==============================

Investigating social prediction using eye tracking and behavioral performance

### Installing the Required Python Packages

This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management.

Install pipenv globally if it's not already installed:

    $ brew install pipenv

Navigate to the top-level directory in `social_eye` and install the packages from the `Pipfile.lock`.
This will automatically create a new virtual environment for you and install all requirements using the correct version of python.

    $ pipenv install

## Activating the virtual environment:

    $ pipenv shell

> NOTE: To deactivate the virtual environment when you are done working, simply type `exit`

## Activating the ipykernel

    $ python -m ipykernel install --user --name social_eye

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── eyetracking    <- eyetracking data
    │   └── behavior       <- behavioral data
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── Pipfile            <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pipenv install -e .) so src can be imported
    ├── analysis_code                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    |   |
    │   |── constants.py   <- Module to set default directories for project
    │   │
    │   ├── scripts        <- Scripts to transfer data from savio
    │   │   └── data_transfer.py
    |   |
    │   │── visualize_eye.py  <- Module to visualize eyetracking
    │   │
    │   └── visualize_behavior.py  <- Module to visualize behavior
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------