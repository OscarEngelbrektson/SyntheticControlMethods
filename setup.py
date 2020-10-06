from setuptools import setup, find_packages

install_requires = [
        'numpy',
        'scipy==1.4.1',
        'pandas',
        'cvxpy',
        'matplotlib>=2.2.3',
        'jinja2>=2.10'
    ]

setup(name='SyntheticControl', version='0.0.1', packages=find_packages(), install_requires=install_requires)
'''
setup(
    name='pysynth',
    version=_version['__version__'],
    author='Oscar Engelbrektson',
    author_email='engelbrektson.oscar@gmail.com',
    url='https://github.com/OscarEngelbrektson/SyntheticControl',
    description= "Python version of Google's Causal Impact model",
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=packages,
    include_package_data=True,
    install_requires=install_requires,
    tests_require=tests_require,
    setup_requires=setup_requires,
    extras_require=extras_require,
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
        'Topic :: Scientific',
        'Topic :: Econometrics',
        'Topic :: Causal Inference',
        'Topic :: Impact Evaluation',
    ],
    cmdclass={'test': PyTest},
    test_suite='tests'
)
'''