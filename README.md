action_prediction
==============================

## Project Description

This project is part of the final project of my PhD thesis. I collected 12 hours of varsity soccer players taking penalty shots at 16 unique targets (incorporating a difficulty manipulation).Then, I designed an experiment to test human participants on their ability to predict the direction of the ball (left or right) and I used behavioral performance and eye-tracking as metrics of learning. Next, I used deeplabcut (http://www.mackenziemathislab.org/deeplabcut) to perform markerless labeling of the players' effectors, and I trained a model on these features to determine accuracy of the model relative to human learners. This project is still a work in progress so the final model is still undetermined (hopefully soon!). 

> NOTE: I also collected 15 hours of actors engaging in social greetings: hugging, high-fiving, shaking hands in order to make action predictions in a social context. I'm currently analyzing these data. 

### Markerless labeling of soccer player taking penalty shot
I used deeplabcut to label in real-time the effectors (shoulder, knee, arm, leg etc.) of 3 varsity soccer players as they took directed penalty shots at 16 unique targets. 

[![Markerless labeling of soccer players](https://res.cloudinary.com/marcomontalbano/image/upload/v1633201853/video_to_markdown/images/google-drive--1UYxtM0v1wjqGs36ATkP9FEOVmeedJjzz-c05b58ac6eb4c4700831b2b3070cd403.jpg)](https://drive.google.com/file/d/1UYxtM0v1wjqGs36ATkP9FEOVmeedJjzz/view?usp=sharing "Markerless labeling of soccer players")

### Plot poses tracking the player's body movements
The relative position of the effectors are plotted in 2D as the soccer player runs and takes a penalty shot. X and Y positions are given in pixels. 

![trajectory_filtered](https://user-images.githubusercontent.com/28731306/135729298-933f530f-c79b-47ba-af74-f31b00270cf5.png)

### Research Assistants: Installing the Required Python Packages

This project uses [`pipenv`](https://github.com/pypa/pipenv) for virtual environment and python package management.
Make sure that you have installed `pyenv` and `pipenv`.

Install `pyenv` using Homebrew:

    $ brew update
    $ brew install pyenv

Add `pyenv init` to your shell:

    $ echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bash_profile
    $ source ~/.bash_profile

Install the required version of python:

    $ pyenv install 3.7.0

Ensure pipenv is installed globally:

    $ brew install pipenv

Navigate to the top-level directory in `action_prediction` and install the packages from the `Pipfile.lock`.
This will automatically create a new virtual environment for you and install all requirements using the correct version of python.

    $ pipenv install

Activating the virtual environment:

    $ pipenv shell

> NOTE: To deactivate the virtual environment when you are done working, simply type `exit`

Activating the ipykernel

    $ python -m ipykernel install --user --name action_prediction
    
Data Organization
------------
    ├── data
    │   ├── eyetracking    <- eyetracking data `group_eyetracking.csv`
    │   └── behavior       <- behavioral data `group_behavior.csv`


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
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
    ├── analysis_code      <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    |   |
    │   |── constants.py   <- Module to set default directories for project
    │   │
    |   |──scripts         <- folder that contains scripts
    |   |
    │   │── visualize_eye.py  <- Module to visualize eyetracking
    │   │
    │   └── visualize_behavior.py  <- Module to visualize behavior
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------
