#! /usr/bin/env python3

import sys
import time
import timeit
from pandas import DataFrame, Series
import random
import array
import pickle
import itertools
import datetime
import argparse

import pyroaring
from pyroaring import BitMap


def get_list(size, density):
    '''Return a random (uniform) list with the desired number of elements and density.'''
    universe_size = int(size/density)
    return random.sample(range(universe_size), size)


def get_range(size, density):
    '''Return a range with the desired number of elements and density.'''
    universe_size = int(size/density)
    return range(0, universe_size, int(1/density))


import_str = 'import array, pickle; from __main__ import %s' % (','.join(
    ['get_list', 'get_range', 'random', 'BitMap']))


def run_exp(stmt, setup, number):
    setup = '%s ; %s' % (import_str, setup)
    try:
        return timeit.timeit(stmt=stmt, setup=setup, number=number)/number
    except Exception as e:
        print(e)
        return float('nan')


constructor = 'x={class_name}(values)'
simple_setup_constructor = 'x={class_name}(get_list({size}, {density}));val=random.randint(0, x.max());N=len(x)'
double_setup_constructor = 'x={class_name}(get_list({size}, {density})); y={class_name}(get_list({size}, {density}))'
equal_setup_constructor = 'l=get_list({size}, {density});x={class_name}(l); y={class_name}(l)'
experiments = [
    # Constructors
    ('range constructor', ('values=get_range({size}, {density})', constructor)),
    ('ordered list constructor', ('values=get_list({size}, {density}); values.sort()', constructor)),
    ('list constructor', ('values=get_list({size}, {density})', constructor)),
    ('ordered array constructor', ('l=get_list({size}, {density}); l.sort(); values=array.array("I", l)', constructor)),
    ('array constructor', ('values=array.array("I", get_list({size}, {density}))', constructor)),
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
    ('selection', (simple_setup_constructor, 'x[int(N/2)]')),
    ('contiguous slice', (simple_setup_constructor, 'x[int(N/4):int(3*N/4):1]')),
    ('slice', (simple_setup_constructor, 'x[int(N/4):int(3*N/4):3]')),
    ('small slice', (simple_setup_constructor, 'x[int(N/100):int(3*N/100):3]')),
]
exp_dict = dict(experiments)


def run(cls, op, size, density, nb_calls):
    cls_name = cls.__name__
    setup = exp_dict[op][0].format(class_name=cls_name, size=size, density=density)
    stmt = exp_dict[op][1].format(class_name=cls_name)
    result = run_exp(stmt=stmt, setup=setup, number=nb_calls)
    return result


def run_all(nb_calls, nb_runs, sizes, densities, experiments):
    main_start = time.time()
    columns = {
        'operation': Series([], dtype='str'),
        'density': Series([], dtype='float'),
        'size': Series([], dtype='int'),
        'time': Series([], dtype='float'),
        'timestamp': Series([], dtype='str'),
    }
    df = DataFrame(columns)
    experiments = list(itertools.product(experiments, sizes, densities))
    for run_id in range(nb_runs):
        sys.stderr.write(f'Run {run_id+1:2d}/{nb_runs:2d}\n')
        start = time.time()
        random.shuffle(experiments)
        for (op, _), size, density in experiments:
            # sys.stderr.write(f'    experiment: {op.rjust(40)} | {size:9d} | {density:.4f}\n')
            now = str(datetime.datetime.now())
            result = {'timestamp': now, 'operation': op, 'density': density, 'size': size}
            result['time'] = run(BitMap, op, size, density, nb_calls)
            df.loc[len(df.index)] = result
        stop = time.time()
        run_time = stop - start
        sys.stderr.write(f'    Done in {run_time:.1f} seconds\n')
        if run_id < nb_runs-1:
            sys.stderr.write(f'    Estimated remaining time: {(stop-main_start)/(run_id+1)*(nb_runs-run_id-1):.1f} seconds\n')
    df['nb_calls'] = nb_calls
    sys.stderr.write(f'Terminated in {stop-main_start:.1f} seconds\n')
    return df


def main():
    parser = argparse.ArgumentParser(description='Benchmark for pyroaring.')
    parser.add_argument('--nb_runs', type=int, default=10,
            help='Number of times to run each series of calls.')
    parser.add_argument('--nb_calls', type=int, default=10,
            help='Number of times to run each experiment in one run.')
    parser.add_argument('--sizes', type=int, default=[1000000], nargs='+',
            help='List of sizes to use for the bitmaps.')
    parser.add_argument('--densities', type=float, default=[0.01, 0.1, 0.99], nargs='+',
            help='List of sizes to use for the bitmaps.')
    parser.add_argument('--add_versions', action='store_true',
            help='Add the versions of pyroaring and croaring to the output file.')
    parser.add_argument('output', type=argparse.FileType('w'))
    args = parser.parse_args()
    d = min(args.densities)
    if d < 0:
        parser.error(f'Invalid density of {d}, must be positive.')
    d = max(args.densities)
    if d > 1:
        parser.error(f'Invalid density of {d}, must be lower than 1.')
    df = run_all(nb_calls=args.nb_calls,
                 nb_runs=args.nb_runs,
                 sizes=args.sizes,
                 densities=args.densities,
                 experiments=experiments)
    if args.add_versions:
        df['pyroaring_version'] = pyroaring.__version__
        df['croaring_version'] = pyroaring.__croaring_version__
    df.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
