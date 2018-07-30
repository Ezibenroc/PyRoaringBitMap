#! /usr/bin/env python3

from setuptools import setup
from setuptools.extension import Extension
from distutils.sysconfig import get_config_vars
import os
import sys
import subprocess
import platform

VERSION = '0.2.4'
PKG_DIR = 'pyroaring'

PLATFORM_WINDOWS = (platform.system() == 'Windows')
PLATFORM_MACOSX = (platform.system() == 'Darwin')


def chdir(func, directory):
    old_dir = os.getcwd()
    os.chdir(directory)
    res = func()
    os.chdir(old_dir)
    return res


def run(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        sys.exit('Error with the command %s.\n' % ' '.join(args))
    return stdout.decode('ascii').strip()


def git_version():
    return run(['git', 'rev-parse', 'HEAD'])


def git_tag():
    return run(['git', 'describe', '--always', '--dirty'])


def write_version(filename, version_dict):
    with open(filename, 'w') as f:
        for version_name in version_dict:
            f.write('%s = "%s"\n' % (version_name, version_dict[version_name]))


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

USE_CYTHON = os.path.exists(os.path.join(PKG_DIR, 'pyroaring.pyx'))
if USE_CYTHON:
    print('Building pyroaring from Cython sources.')
    from amalgamation import amalgamate
    amalgamate(PKG_DIR)
    from Cython.Build import cythonize
    ext = 'pyx'
    write_version(os.path.join(PKG_DIR, 'version.pxi'), {
        '__version__': VERSION,
        '__git_version__': git_version(),
        '__croaring_version__': chdir(git_tag, 'CRoaring'),
        '__croaring_git_version__': chdir(git_version, 'CRoaring'),
    })
else:
    print('Building pyroaring from C sources.')
    ext = 'cpp'

if PLATFORM_WINDOWS:
    compile_args = []
else:
    compile_args = ['-D__STDC_LIMIT_MACROS', '-D__STDC_CONSTANT_MACROS']
    if not PLATFORM_MACOSX:
        compile_args.append('-std=c99')
    if 'DEBUG' in os.environ:
        compile_args.extend(['-O0', '-g'])
    else:
        compile_args.append('-O3')
    if 'ARCHI' in os.environ:
        compile_args.extend(['-march=%s' % os.environ['ARCHI']])
    else:
        compile_args.append('-march=native')

filename = os.path.join(PKG_DIR, 'pyroaring.%s' % ext)
pyroaring = Extension('pyroaring',
                      sources=[filename, os.path.join(PKG_DIR, 'roaring.c')],
                      extra_compile_args=compile_args,
                      )
if USE_CYTHON:
    pyroaring = cythonize(pyroaring, compiler_directives={'binding': True})
else:
    pyroaring = [pyroaring]

setup(
    name='pyroaring',
    ext_modules=pyroaring,
    version=VERSION,
    description='Fast and lightweight set for unsigned 32 bits integers.',
    long_description=long_description,
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
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
