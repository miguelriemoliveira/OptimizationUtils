
import os
from setuptools import setup, find_packages

# Utility function to read the content of a file file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "OptimizationUtils",
    version = "1.0.0",
    install_requires=[],
    author = "Miguel Oliveira",
    author_email = "mike@todo.todo",
    description = ("A set of utilities for using the python scipy optimizer functions"),
    license = "GPLv3",
    keywords = "optimization",
    url = "https://github.com/miguelriemoliveira/OptimizationUtils",
    packages=find_packages(exclude=['test*']),
    package_dir={'OptimizationUtils': 'OptimizationUtils'},
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 2.7",
        "Topic :: Utilities",
    ],
)
