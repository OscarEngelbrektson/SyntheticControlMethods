from setuptools import setup, find_packages

install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'cvxpy',
        'matplotlib>=2.2.3',
        'jinja2>=2.10'
    ]

setup(name='SyntheticControl', version='0.0.1', packages=find_packages(), install_requires=install_requires)