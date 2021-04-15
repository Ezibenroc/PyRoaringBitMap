import os
import sys
import datetime
import re

CROARING_DIR = 'CRoaring'
SRC_DIR = os.path.join(CROARING_DIR, 'src')
INCLUDE_DIR = os.path.join(CROARING_DIR, 'include', 'roaring')
SRC_FILE = 'roaring.c'
INCLUDE_FILE = 'roaring.h'

LICENSE_TXT = '''Copyright %s The CRoaring authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Github repository: https://github.com/RoaringBitmap/CRoaring/
Official website : http://roaringbitmap.org/''' % datetime.date.today().year


def find_src_files(src_dir=SRC_DIR):
    src_files = []
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith(".c"):
                src_files.append(os.path.join(root, file))
    src_files.sort()
    return src_files


def find_include_files(file_list, include_dir):
    include_files = [os.path.join(INCLUDE_DIR, f) for f in file_list]
    return include_files


def find_include_private_files(include_dir=INCLUDE_DIR):
    include_files = [
        'isadetection.h',
        'portability.h',
        'containers/perfparameters.h',
        'containers/container_defs.h',
        'array_util.h',
        'utilasm.h',
        'bitset_util.h',
        'containers/array.h',
        'containers/bitset.h',
        'containers/run.h',
        'containers/convert.h',
        'containers/mixed_equal.h',
        'containers/mixed_subset.h',
        'containers/mixed_andnot.h',
        'containers/mixed_intersection.h',
        'containers/mixed_negation.h',
        'containers/mixed_union.h',
        'containers/mixed_xor.h',
        'containers/containers.h',
        'roaring_array.h',
        'misc/configreport.h',
    ]
    return find_include_files(include_files, include_dir)


def find_include_public_files(include_dir=INCLUDE_DIR):
    include_files = [
        'roaring_version.h',
        'roaring_types.h',
        'roaring.h',
    ]
    return find_include_files(include_files, include_dir)


def check_file_list(file_list):
    error = False
    for file in file_list:
        if not os.path.isfile(file):
            error = True
            sys.stderr.write('File not found: %s\n' % file)
    if error:
        sys.exit(1)
    return file_list


def amalgamate_file(file_list, output_file, license_txt=LICENSE_TXT, additional_txt=None):
    regex_1 = '#include\s*"[a-zA-Z_/]+.h"\s*'
    regex_2 = '#include\s*<roaring/[a-zA-Z_/]+.h>\s*'
    regex = re.compile('(%s)|(%s)' % (regex_1, regex_2))
    with open(output_file, 'w') as output_f:
        output_f.write('/* File automatically generated on %s. */\n\n' %
                       datetime.date.today())
        if license_txt:
            output_f.write('/*\n%s\n*/\n\n' % license_txt)
        if additional_txt:
            output_f.write('%s\n\n' % additional_txt)
        for input_file in file_list:
            output_f.write('/* Begin file %s */\n' % input_file)
            with open(input_file, 'r') as input_f:
                for line in input_f:
                    match = regex.match(line)
                    if match:
                        line = ''
                    output_f.write(line)
            output_f.write('/* End file %s */\n' % input_file)


def amalgamate(target_dir):
    src_files = check_file_list(['custom_roaring.c'] + find_src_files())
    include_public_files = check_file_list(find_include_public_files() + ['custom_roaring.h'])
    include_private_files = check_file_list(find_include_private_files())
    target_src = os.path.join(target_dir, SRC_FILE)
    target_include = os.path.join(target_dir, INCLUDE_FILE)
    amalgamate_file(include_private_files + src_files, target_src,
                    additional_txt='#include "%s"' % INCLUDE_FILE)
    amalgamate_file(include_public_files, target_include)
