import os
import sys
import os.path
from subprocess import Popen
import shutil

PYROARING_TAG = 'pyroaring_installation'
BLUE_STR = '\033[1m\033[94m'
END_STR = '\033[0m'

def print_msg(msg):
    print('%s%s%s' % (BLUE_STR, msg, END_STR))

class SubProcessError(Exception):
    pass

def error(msg):
    sys.stderr.write('ERROR: %s\n' % msg)
    sys.exit(1)

syntax_msg = 'python[3] -m pyroaring <install|uninstall> [--user]'

def syntax():
    sys.stderr.write('Syntax: %s\n' % syntax_msg)
    sys.exit(1)

def run_command(args):
    print_msg(' '.join(args))
    process = Popen(args)
    if process.wait() != 0:
        error('with command: %s' % ' '.join(args))
    print('')

def get_sources(target_dir):
    if not os.path.exists(target_dir):
        run_command(['git', 'clone', '--depth', '1', 'https://github.com/RoaringBitmap/CRoaring.git', target_dir])
    else:
        old_path = os.getcwd()
        os.chdir(target_dir)
        run_command(['git', 'pull'])
        os.chdir(old_path)

def build_library(sources_dir, lib_name):
    old_path = os.getcwd()
    os.chdir(sources_dir)
    run_command(['bash', 'amalgamation.sh']) # TODO remove bash dependency
    run_command(['gcc', '-march=native', '-O3', '-std=c11', '-shared', '-o', lib_name, '-fPIC', 'roaring.c'])
    os.chdir(old_path)

def fetch_and_build(sources_dir, lib_name):
    get_sources(sources_dir)
    build_library(sources_dir, lib_name)

def install(sources_dir, lib_name):
    fetch_and_build(sources_dir, lib_name)
    # TODO: write cross-platform code
    try:
        print_msg('Copying library files in /usr/local/lib and /usr/local/include/roaring')
        shutil.copy(os.path.join(sources_dir, lib_name), '/usr/local/lib/')
        os.makedirs('/usr/local/include/roaring')
        shutil.copy(os.path.join(sources_dir, 'roaring.h'), '/usr/local/include/roaring')
    except PermissionError:
        error('permission denied. This might require superuser priviledges (try runing the command with "sudo").')

def get_shell_file():
    shell = os.path.split(os.environ['SHELL'])[1]
    try:
        shell_file = {
                'zsh' :  '.zshrc',
                'bash' : '.bashrc',
            }[shell]
    except KeyError:
        error('Unknown shell "%s"' % shell)
    home_dir = os.environ['HOME']
    config_file = os.path.join(home_dir, shell_file)
    return config_file

def install_local(sources_dir, lib_name):
    fetch_and_build(sources_dir, lib_name)
    config_file = get_shell_file()
    print_msg('Adding an entry in %s' % config_file)
    with open(config_file, 'a') as f:
        f.write('export LD_LIBRARY_PATH="%s:$LD_LIBRARY_PATH" # %s\n' % (sources_dir, PYROARING_TAG))

def uninstall(lib_name):
    try:
        print_msg('Removing library files from /usr/local/lib and /usr/local/include/roaring')
        os.remove(os.path.join('/usr/local/lib/', lib_name))
        os.remove('/usr/local/include/roaring/roaring.h')
        os.removedirs('/usr/local/include/roaring/')
    except PermissionError:
        error('permission denied. This might require superuser priviledges (try runing the command with "sudo").')

def uninstall_local():
    config_file = get_shell_file()
    print_msg('Removing pyroaring entries from %s' % config_file)
    with open(config_file, 'r') as f:
        lines = f.readlines()
    with open(config_file, 'w') as f:
        for line in lines:
            if PYROARING_TAG not in line:
                f.write(line)

def main_install():
    if not 2 <= len(sys.argv) <= 3:
        syntax()
    sources_dir = os.path.join(os.getcwd(), 'croaring')
    lib_name = 'libroaring.so'
    if sys.argv[1] == 'install':
        if len(sys.argv) == 2:
            install(sources_dir, lib_name)
        elif sys.argv[2] == '--user':
            install_local(sources_dir, lib_name)
        else:
            syntax()
    elif sys.argv[1] == 'uninstall':
        if len(sys.argv) == 2:
            uninstall(lib_name)
        elif sys.argv[2] == '--user':
            uninstall_local()
        else:
            syntax()
    else:
        syntax()

if __name__ == '__main__':
    main_install()
