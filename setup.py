from setuptools import setup, find_packages

import os
import re
import sys
from codecs import open

install_requires = [
        'numpy>=1.17',
        'scipy==1.4.1',
        'pandas',
        'cvxpy',
        'matplotlib>=2.2.3',
        'jinja2>=2.10'
    ]

#Get version
here = os.path.abspath(os.path.dirname(__file__))
_version = {}
_version_path = os.path.join(here, 'SyntheticControlMethods', '__version__.py')
with open(_version_path, 'r', 'utf-8') as f:
    exec(f.read(), _version)

#Get README.md for long description
with open('README.md', 'r', 'utf-8') as f:
    readme = f.read()

setup(
    name='SyntheticControlMethods',
    version=_version['__version__'],
    author='Oscar Engelbrektson',
    author_email='oscar.engelbrektson@gmail.com',
    url='https://github.com/OscarEngelbrektson/SyntheticControlMethods',
    download_url='https://github.com/OscarEngelbrektson/SyntheticControlMethods',
    description= "A Python package for causal inference using various Synthetic Control Methods",
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    include_package_data=True,
    install_requires=install_requires,
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
)