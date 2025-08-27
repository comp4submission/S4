import os
import math
import argparse
import functools
import subprocess

from tqdm import tqdm
from utils import generator
from multiprocessing import Pool


def batch_process(file_list, args):
    for f in file_list:
        subprocess.call([args.ida_path, '-A', '-B', f], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return 0


def get_args():
    arg_parser = argparse.ArgumentParser(description='This script is to generate idb from binaries in batches')
    arg_parser.add_argument('--bin_path', type=str, default='/mnt/ssd1/anonymous/wo_debug/x86_64')
    arg_parser.add_argument('--ida_path', type=str, default='/opt/idapro-8.3/ida64')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count()//2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=64, help='workload per group')

    args = arg_parser.parse_args()
    return args


def main():

    args = get_args()

    # check whether the key files exist
    if not os.path.exists(args.bin_path):
        print('[ERROR]', 'Binary Path Does Not Exist}')
        exit(0)
    if not os.path.exists(args.ida_path):
        print('[ERROR]', 'IDA Path Does Not Exist}')
        exit(0)

    # get all binaries
    bins = []
    for bin_dir in os.listdir(args.bin_path):
        for binary in os.listdir(os.path.join(args.bin_path, bin_dir)):
            bins.append(os.path.join(args.bin_path, bin_dir, binary))

    print('[INFO]', f'# of bins: {len(bins)}')

    # dispatch task to each core
    group_count = math.ceil(len(bins) / args.workload)

    batch_process_partial = functools.partial(batch_process, args=args)
    with Pool(processes=args.core) as pool:
        for _ in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(bins, args.workload)),
                total=group_count):
            pass
    print('[INFO]', 'Disassembly Completed')

    # statistics of disassembly
    idbs = []
    asms = []
    for bin_dir in os.listdir(args.bin_path):
        for file in os.listdir(os.path.join(args.bin_path, bin_dir)):
            if file.endswith('.i64'):
                idbs.append(os.path.join(args.bin_path, bin_dir, file))
            elif file.endswith('.asm'):
                asms.append(os.path.join(args.bin_path, bin_dir, file))

    print('[INFO]', f'# of bins: {len(bins)}')
    print('[INFO]', f'# of idbs: {len(idbs)}')
    print('[INFO]', f'# of asms: {len(asms)}')


if __name__ == '__main__':
    main()

