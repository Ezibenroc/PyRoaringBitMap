#! /usr/bin/env python3

import sys
import random
import timeit

from pandas import Series, DataFrame

try:
    import tabulate
    has_tabulate = True
except ImportError:
    has_tabulate = False
    sys.stderr.write('Warning: could not import tabulate\n')
    sys.stderr.write('         see https://bitbucket.org/astanin/python-tabulate\n')
from pyroaring import BitMap, BitMap64

classes = {'set': set, 'pyroaring (32 bits)': BitMap, 'pyroaring (64 bits)': BitMap64, }
nb_exp = 30
size = int(1e6)
density = 0.125
universe_size = int(size / density)

try:
    from roaringbitmap import RoaringBitmap
    classes['roaringbitmap'] = RoaringBitmap
except ImportError:
    sys.stderr.write('Warning: could not import roaringbitmap\n')
    sys.stderr.write('         see https://github.com/andreasvc/roaringbitmap/\n')

try:
    from sortedcontainers.sortedset import SortedSet
    classes['sortedcontainers'] = SortedSet
except ImportError:
    sys.stderr.write('Warning: could not import sortedcontainers\n')
    sys.stderr.write('         see https://github.com/grantjenks/sorted_containers\n')

try:
    from croaring import BitSet
    classes['python-croaring'] = BitSet
except ImportError:
    sys.stderr.write('Warning: could not import croaring\n')
    sys.stderr.write('         see https://github.com/sunzhaoping/python-croaring\n')

import_str = 'import array, pickle; from __main__ import %s' % (','.join(
    ['get_list', 'get_range', 'random', 'size', 'universe_size']
    + [cls.__name__ for cls in classes.values() if cls is not set]))


def run_exp(stmt, setup, number):
    setup = '%s ; %s' % (import_str, setup)
    try:
        return timeit.timeit(stmt=stmt, setup=setup, number=number) / number
    except Exception:
        return float('nan')


def get_range():
    r = (0, universe_size, int(1 / density))
    try:
        return xrange(*r)
    except NameError:
        return range(*r)


def get_list():
    return random.sample(range(universe_size), size)


constructor = 'x={class_name}(values)'
simple_setup_constructor = 'x={class_name}(get_list());val=random.randint(0, universe_size)'
double_setup_constructor = 'x={class_name}(get_list()); y={class_name}(get_list())'
equal_setup_constructor = 'l=get_list();x={class_name}(l); y={class_name}(l)'
experiments = [
    # Constructors
    ('range constructor', ('values=get_range()', constructor)),
    ('ordered list constructor', ('values=get_list(); values.sort()', constructor)),
    ('list constructor', ('values=get_list()', constructor)),
    ('ordered array constructor', ('l=get_list(); l.sort(); values=array.array("I", l)', constructor)),
    ('array constructor', ('values=array.array("I", get_list())', constructor)),
    # Simple operations
    ('element addition', (simple_setup_constructor, 'x.add(val)')),
    ('element removal', (simple_setup_constructor, 'x.discard(val)')),
    ('membership test', (simple_setup_constructor, 'val in x')),
    # Binary operations
    ('union', (double_setup_constructor, 'z=x|y')),
    ('intersection', (double_setup_constructor, 'z=x&y')),
    ('difference', (double_setup_constructor, 'z=x-y')),
    ('symmetric diference', (double_setup_constructor, 'z=x^y')),
    ('equality test', (equal_setup_constructor, 'x==y')),
    ('subset test', (equal_setup_constructor, 'x<=y')),
    # Export
    ('conversion to list', (simple_setup_constructor, 'list(x)')),
    ('pickle dump & load', (simple_setup_constructor, 'pickle.loads(pickle.dumps(x, protocol=pickle.HIGHEST_PROTOCOL))')),
    ('"naive" conversion to array', (simple_setup_constructor, 'array.array("I", x)')),
    ('"optimized" conversion to array', (simple_setup_constructor, 'x.to_array()')),
    # Items
    ('selection', (simple_setup_constructor, 'x[int(size/2)]')),
    ('contiguous slice', (simple_setup_constructor, 'x[int(size/4):int(3*size/4):1]')),
    ('slice', (simple_setup_constructor, 'x[int(size/4):int(3*size/4):3]')),
    ('small slice', (simple_setup_constructor, 'x[int(size/100):int(3*size/100):3]')),
]
exp_dict = dict(experiments)


def run(cls, op):
    cls_name = classes[cls].__name__
    setup = exp_dict[op][0].format(class_name=cls_name)
    stmt = exp_dict[op][1].format(class_name=cls_name)
    result = run_exp(stmt=stmt, setup=setup, number=nb_exp)
    return result


def run_all():
    all_results = []
    for op, _ in experiments:
        sys.stderr.write('experiment: %s\n' % op)
        result = {'operation': op}
        for cls in random.sample(list(classes), len(classes)):
            result[cls] = run(cls, op)
        all_results.append(result)
    return DataFrame(all_results).sort_index(axis=1)


if __name__ == '__main__':
    df = run_all()
    print()
    if has_tabulate:
        print(tabulate.tabulate(df, headers='keys', tablefmt='rst', showindex='never', floatfmt=".2e"))
    else:
        print(df)
