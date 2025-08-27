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
        subprocess.call([args.strip_path, '-g', '--remove-section=.comment', f], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return 0


def get_args():
    arg_parser = argparse.ArgumentParser(description='This script is to strip binaries in batches.'
                                                     'The targeted binary folder should follow the new arrangement')
    arg_parser.add_argument('--bin_path', type=str, default='/mnt/ssd1/anonymous/wo_debug/x86_64')
    arg_parser.add_argument('--strip_path', type=str, default='/usr/bin/strip')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count()//2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=256, help='workload per group')

    args = arg_parser.parse_args()
    return args


def main():

    args = get_args()

    # check whether the key files exist
    if not os.path.exists(args.bin_path):
        print('[ERROR]', 'Binary Path Does Not Exist}')
        exit(0)
    if not os.path.exists(args.strip_path):
        print('[ERROR]', 'Strip Path Does Not Exist}')
        exit(0)

    # get all binaries
    binaries = []
    for bin_dir in os.listdir(args.bin_path):
        for binary in os.listdir(os.path.join(args.bin_path, bin_dir)):
            binaries.append(os.path.join(args.bin_path, bin_dir, binary))

    print('[INFO]', 'Number of Binaries:', len(binaries))

    # dispatch task to each core
    group_count = math.ceil(len(binaries) / args.workload)

    batch_process_partial = functools.partial(batch_process, args=args)
    with Pool(processes=args.core) as pool:
        for _ in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(binaries, args.workload)),
                total=group_count):
            pass
    print('[INFO]', 'Stripping Completed')


if __name__ == '__main__':
    main()

