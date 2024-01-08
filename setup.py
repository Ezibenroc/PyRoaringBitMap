#! /usr/bin/env python3

from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_config_vars
import os
import sys
import subprocess
import platform


PKG_DIR = 'pyroaring'

PLATFORM_WINDOWS = (platform.system() == 'Windows')
PLATFORM_MACOSX = (platform.system() == 'Darwin')

# Read version file from the src
with open("pyroaring/version.pxi") as fp:
    exec(fp.read())
    VERSION = __version__


# Remove -Wstrict-prototypes option
# See http://stackoverflow.com/a/29634231/4110059
if not PLATFORM_WINDOWS:
    cfg_vars = get_config_vars()
    for key, value in cfg_vars.items():
        if type(value) == str:
            cfg_vars[key] = value.replace("-Wstrict-prototypes", "")

try:
    with open('README.rst') as f:
        long_description = ''.join(f.readlines())
except (IOError, ImportError, RuntimeError):
    print('Could not generate long description.')
    long_description = ''


if PLATFORM_WINDOWS:
    pyroaring_module = Extension(
        'pyroaring',
        sources=[os.path.join(PKG_DIR, 'pyroaring.pyx'), os.path.join(PKG_DIR, 'roaring.c')],
        language='c++'
    )
    libraries = None
else:
    compile_args = ['-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS', '-D _GLIBCXX_ASSERTIONS']
    if PLATFORM_MACOSX:
        compile_args.append('-mmacosx-version-min=10.14')
    if 'DEBUG' in os.environ:
        compile_args.extend(['-O0', '-g'])
    else:
        compile_args.append('-O3')
    if 'ARCHI' in os.environ:
        if os.environ['ARCHI'] != "generic":
            compile_args.extend(['-march=%s' % os.environ['ARCHI']])
    # The '-march=native' flag is not universally allowed. In particular, it
    # will systematically fail on aarch64 systems (like the new Apple M1 systems). It
    # also creates troubles under macOS with pip installs and requires ugly workarounds.
    # The best way to handle people who want to use -march=native is to ask them
    # to pass ARCHI=native to their build process.
    #else:
    #    compile_args.append('-march=native')

    pyroaring_module = Extension(
        'pyroaring',
        sources=[os.path.join(PKG_DIR, 'pyroaring.pyx')],
        extra_compile_args=compile_args + ["-std=c++11"],
        language='c++'
    )

    # Because we compile croaring with a c compiler with sometimes incompatible arguments,
    # define croaring compilation with an extra argument for the c11 standard, which is
    # required for atomic support.
    croaring = (
        'croaring',
        {
            'sources': [os.path.join(PKG_DIR, 'roaring.c')],
            "extra_compile_args": compile_args + ["-std=c11"]
        },
    )
    libraries = [croaring]

setup(
    name='pyroaring',
    ext_modules=[pyroaring_module],
    libraries=libraries,
    version=VERSION,
    description='Fast and lightweight set for unsigned 32 bits integers.',
    long_description=long_description,
    setup_requires=['cython'],
    url='https://github.com/Ezibenroc/PyRoaringBitMap',
    author='Tom Cornebize',
    author_email='tom.cornebize@gmail.com',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
