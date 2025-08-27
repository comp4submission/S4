# python3
import os
import math
import argparse
import functools

from tqdm import tqdm
from typing import Any
from multiprocessing import Pool
from elftools.elf.elffile import ELFFile


def is_elf_file(file_path: str):
    with open(file_path, 'rb') as f:
        return f.read(4) == b'\x7fELF'


def generator(full_list: list[Any], sublist_length: int) -> list[Any]:
    group_count = math.ceil(len(full_list) / sublist_length)

    for idx in range(group_count):
        start = sublist_length * idx
        end = min(start + sublist_length, len(full_list))

        yield full_list[start:end]


def batch_process(file_list, args):

    for f in file_list:

        # f: /home/anonymous/compass/wo_debug/x86_64/xorriso-1.5.4_xorriso/gcc-9.4.0_Os
        # dir: gcc-9.4.0
        # filename: xorriso-1.5.4_xorriso_Os

        compiler, opt = os.path.basename(f).split('_', maxsplit=1)
        out_filename = f'{os.path.basename(os.path.dirname(f))}_{opt}.bin'

        out_path = os.path.join(args.out_path, compiler)
        # out_path = os.path.join(args.out_path, opt)

        os.makedirs(out_path, exist_ok=True)

        extract_text_section(f, os.path.join(out_path, out_filename))
    return 0


def extract_text_section(in_file, out_file):

    with open(in_file, 'rb') as f:
        elffile = ELFFile(f)
        for section in elffile.iter_sections():
            if section.name.lower() == '.text':
                with open(out_file, 'wb') as o:
                    o.write(section.data())
                break


def get_args():
    arg_parser = argparse.ArgumentParser(description='This script is to generate dataset for o-glassesX')
    arg_parser.add_argument('--in_path', type=str, default='/dev/shm/split_dataset/train')
    arg_parser.add_argument('--out_path', type=str, default='/dev/shm/ogx_dataset/compiler/train')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count()//2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=64, help='workload per group')

    args = arg_parser.parse_args()
    return args


def main():

    args = get_args()

    # check whether the key files exist
    if not os.path.exists(args.in_path):
        print('[ERROR]', 'Binary Path Does Not Exist}')
        exit(0)

    os.makedirs(args.out_path, exist_ok=True)

    # get all binaries
    bins = []
    for bin_dir in os.listdir(args.in_path):
        for file in os.listdir(os.path.join(args.in_path, bin_dir)):
            if is_elf_file(os.path.join(args.in_path, bin_dir, file)):
                bins.append(os.path.join(args.in_path, bin_dir, file))

    print('[INFO]', f'# of bins: {len(bins)}')

    # dispatch task to each core
    group_count = math.ceil(len(bins) / args.workload)

    batch_process_partial = functools.partial(batch_process, args=args)
    with Pool(processes=args.core) as pool:
        for _ in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(bins, args.workload)),
                total=group_count):
            pass
    print('[INFO]', 'Code Section Extraction Completed')


if __name__ == '__main__':
    main()

