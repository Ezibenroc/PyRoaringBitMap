#!/usr/bin/env python3

from setuptools import setup

setup(
    name='pyroaring',
    version='0.0.3',
    description='Fast and lightweight set for unsigned 32 bits integers.',
    url='https://github.com/Ezibenroc/PyRoaringBitMap',
    author='Tom Cornebize',
    author_email='tom.cornebize@gmail.com',
    license='MIT',
    packages=[
        'pyroaring',
    ],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
