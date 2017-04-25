#! /usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
import os
import sys
os.environ['CC'] = 'cc'

USE_CYTHON = 'PYROARING_CYTHON' in os.environ
if USE_CYTHON:
    print('Building pyroaring from Cython sources.')
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    ext = 'pyx'
else:
    print('Building pyroaring from C sources.')
    ext = 'cpp'
filename = 'pyroaring.%s' % ext
pyroaring = Extension('pyroaring',
                    sources = [filename, 'roaring.c'],
                    extra_compile_args=['-O3', '--std=c99', '-march=native'],
                    language='c++',
                    )
if USE_CYTHON:
    pyroaring = cythonize(pyroaring)
else:
    pyroaring = [pyroaring]

setup(
    name = 'pyroaring',
    ext_modules = pyroaring,
    version='0.0.7',
    description='Fast and lightweight set for unsigned 32 bits integers.',
    url='https://github.com/Ezibenroc/PyRoaringBitMap',
    author='Tom Cornebize',
    author_email='tom.cornebize@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
