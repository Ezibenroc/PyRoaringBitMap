#! /usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
import os
import sys
os.environ['CC'] = 'cc'

def clean_description(descr): # remove the parts with the plots in the README
    start = descr.find('Three interesting plots')
    stop = descr.find('To sum up, both Roaring bitmap implementations')
    assert start != -1 and stop != -1 and start < stop
    return '%s%s' % (descr[:start], descr[stop:])

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
    long_description = clean_description(long_description)
except (IOError, ImportError, RuntimeError):
    print('Could not generate long description.')
    long_description=''

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
                    extra_compile_args=['-O3', '--std=c99', '-march=native', '-Wno-strict-prototypes'],
                    language='c++',
                    )
if USE_CYTHON:
    pyroaring = cythonize(pyroaring)
else:
    pyroaring = [pyroaring]

setup(
    name = 'pyroaring',
    ext_modules = pyroaring,
    version='0.1.0',
    description='Fast and lightweight set for unsigned 32 bits integers.',
    long_description = long_description,
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
