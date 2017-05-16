#! /usr/bin/env python3

import sys
import timeit
from pandas import DataFrame, Series
import pickle
import random
import array
try:
    import tabulate
    has_tabulate = True
except ImportError:
    has_tabulate = False
    sys.stderr.write('Warning: could not import tabulate\n')
    sys.stderr.write('         see https://bitbucket.org/astanin/python-tabulate\n')
from pyroaring import BitMap

classes = {'set': set, 'pyroaring': BitMap}
nb_exp = 30
size = int(1e6)
density = 0.125
universe_size = int(size/density)

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

def run_exp(stmt, setup, number):
    try:
        return timeit.timeit(stmt=stmt, setup=setup, number=number, globals=globals())/number
    except Exception as e:
        return float('nan')

def get_range():
    return range(0, universe_size, int(1/density))

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
    ('pickle dump & load', (simple_setup_constructor, 'pickle.loads(pickle.dumps(x))')),
    ('"naive" conversion to array', (simple_setup_constructor, 'array.array("I", x)')),
    ('"optimized" conversion to array', (simple_setup_constructor, 'x.to_array()')),
    # Items
    ('selection', (simple_setup_constructor, 'x[int(size/2)]')),
    ('slice', (simple_setup_constructor, 'x[int(size/4):int(3*size/4):2]')),
]
exp_dict = dict(experiments)

def run(cls, op):
    cls_name = classes[cls].__name__
    setup = exp_dict[op][0].format(class_name=cls_name)
    stmt = exp_dict[op][1].format(class_name=cls_name)
    result = run_exp(stmt=stmt, setup=setup, number=nb_exp)
    return result

def run_all():
    df = DataFrame({
        'operation': Series([], dtype='str'),
    })
    for cls in sorted(classes):
        df[cls] = Series([], dtype='float')
    for op, _ in experiments:
        sys.stderr.write('experiment: %s\n' % op)
        result = {'operation': op}
        for cls in random.sample(list(classes), len(classes)):
            result[cls] = run(cls, op)
        df=df.append(result, ignore_index=True)
    return df

if __name__ == '__main__':
    df = run_all()
    print()
    if has_tabulate:
        print(tabulate.tabulate(df, headers='keys', tablefmt='rst', showindex='never', floatfmt=".2e"))
    else:
        print(df)
