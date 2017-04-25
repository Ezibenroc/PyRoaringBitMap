#! /usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

sourcefiles = ['pyroaring.pyx']
croaring = cythonize(Extension('pyroaring',
                    sources = ['pyroaring.pyx', 'roaring.c'],
                    extra_compile_args=['-O3', '--std=c11', '-march=native'],
                    language='c++',
                    ))

setup(
    name = 'pyroaring',
    ext_modules = croaring,
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
    ],
)
