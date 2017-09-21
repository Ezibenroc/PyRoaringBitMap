#! /usr/bin/env python3

from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_config_vars
from subprocess import check_output
import os
import sys

# Remove -Wstrict-prototypes option
# See http://stackoverflow.com/a/29634231/4110059
cfg_vars = get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

try:
    with open('README.rst') as f:
        long_description = ''.join(f.readlines())
except (IOError, ImportError, RuntimeError):
    print('Could not generate long description.')
    long_description=''

USE_CYTHON = os.path.exists('pyroaring.pyx')
if USE_CYTHON:
    print('Building pyroaring from Cython sources.')
    check_output(['bash', 'prepare_dist.sh'])
    from Cython.Distutils import build_ext
    from Cython.Build import cythonize
    ext = 'pyx'
else:
    print('Building pyroaring from C sources.')
    ext = 'cpp'

compile_args=['-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS']
if 'DEBUG' in os.environ:
    compile_args.extend(['-O0', '-g'])
else:
    compile_args.append('-O3')
if 'ARCHI' in os.environ:
    compile_args.extend(['-march=%s' % os.environ['ARCHI']])
else:
    compile_args.append('-march=native')

filename = 'pyroaring.%s' % ext
pyroaring = Extension('pyroaring',
                    sources = [filename, 'roaring.cpp'],
                    extra_compile_args=compile_args,
                    language='c++',
                    )
if USE_CYTHON:
    pyroaring = cythonize(pyroaring)
else:
    pyroaring = [pyroaring]

setup(
    name = 'pyroaring',
    ext_modules = pyroaring,
    version='0.1.6',
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
