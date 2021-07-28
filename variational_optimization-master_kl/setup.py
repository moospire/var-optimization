import os
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# optionally if we want to mirror requiremnets.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name = "variational_optimization",
    version = "0.0.1",
    install_requires=[
        'numpy>=1.14.3'
    ],
    author = "Tristan Swedish",
    author_email = "tswedish@mit.edu",
    description = ("Variational Optimization tools"),
    license = "",
    keywords = "",
    url = "",
    packages=['variational_optimization', 'tests'],
    long_description=read('README.md'),
    classifiers=[],
)
