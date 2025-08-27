import os
import math
import argparse
import functools
import subprocess

from tqdm import tqdm
from multiprocessing import Pool


def generator(full_list, sublist_length):
    group_count = math.ceil(len(full_list) / sublist_length)

    for idx in range(group_count):
        start = sublist_length * idx
        end = min(start + sublist_length, len(full_list))

        yield full_list[start:end]


def batch_process(file_list, args):
    for f in file_list:
        subprocess.call([args.ida_path, '-A', f'-S"{args.idapy_path}"', f], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return 0


def get_args():
    arg_parser = argparse.ArgumentParser(description='This script is to run ida python script in batches, '
                                                     'which is based on off-the-shelf idbs.')
    arg_parser.add_argument('--bin_path', type=str, default='/mnt/ssd1/anonymous/wo_debug/x86_64')
    arg_parser.add_argument('--ida_path', type=str, default='/opt/idapro-8.3/ida64')
    arg_parser.add_argument('--idapy_path', type=str, default='/home/anonymous/compass/ida_ext_func_code_norm_inst.py')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count()//2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=32, help='workload per group')

    args = arg_parser.parse_args()
    return args


def main():

    args = get_args()

    # check whether key files exist
    if not os.path.exists(args.bin_path):
        print('[ERROR]', 'Binary Path Does Not Exist}')
        exit(0)
    if not os.path.exists(args.ida_path):
        print('[ERROR]', 'IDA Does Not Exist}')
        exit(0)
    if not os.path.exists(args.idapy_path):
        print('[ERROR]', 'IDAPython Script Does Not Exist}')
        exit(0)

    # get all idbs
    idbs = []
    for bin_dir in os.listdir(args.bin_path):
        for file in os.listdir(os.path.join(args.bin_path, bin_dir)):
            if file.endswith('.i64'):
                idbs.append(os.path.join(args.bin_path, bin_dir, file))

    print('[INFO]', f'# of idbs: {len(idbs)}')

    # dispatch task to each core
    group_count = math.ceil(len(idbs) / args.workload)

    batch_process_partial = functools.partial(batch_process, args=args)
    with Pool(processes=args.core) as pool:
        for _ in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(idbs, args.workload)),
                total=group_count):
            pass
    print('[INFO]', 'IDAPython Script Running Completed')


if __name__ == '__main__':
    main()

