import os
import math
import shutil
import hashlib
import argparse
import functools
import subprocess

from tqdm import tqdm
from multiprocessing import Pool


def is_elf_file(file_path: str):
    with open(file_path, 'rb') as f:
        return f.read(4) == b'\x7fELF'


def generator(full_list, sublist_length):
    group_count = math.ceil(len(full_list) / sublist_length)

    for idx in range(group_count):
        start = sublist_length * idx
        end = min(start + sublist_length, len(full_list))

        yield full_list[start:end]


def sha256(string: str):
    m = hashlib.sha256()
    m.update(string.encode('utf-8'))
    return m.hexdigest()


def batch_process(file_list, args):
    headless_path = os.path.join(args.ghidra_path, 'support', 'analyzeHeadless')
    # ./analyzeHeadless /tmp binkit -overwrite -import /Users/anonymous/PycharmProjects/MyCompass/ls -scriptPath /Users/anonymous/ghidra_scripts -postScript ExtractPcodePseudocode.java
    for f in file_list:
        subprocess.call([headless_path, args.tmp_dir, sha256(f), '-overwrite', '-import', f, '-scriptPath', args.script_path, '-postScript', args.script_name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def get_args():
    arg_parser = argparse.ArgumentParser(description='This script is to run ghidra script in batches')
    arg_parser.add_argument('--bin_path', type=str, default='/mnt/ssd1/anonymous/wo_debug/x86_64')
    arg_parser.add_argument('--ghidra_path', type=str, default='/home/anonymous/ghidra_11.0.3_PUBLIC')
    arg_parser.add_argument('--script_path', type=str, default=os.path.join(os.getcwd(), 'script'))
    # arg_parser.add_argument('--script_name', type=str, default='ExtractPcodePseudocode.java')
    arg_parser.add_argument('--script_name', type=str, default='ExtractOpcodeWeightBasedOnPcode.java')

    arg_parser.add_argument('--tmp_dir', type=str, default='/dev/shm/ghidra_tmp')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count()//2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=12, help='workload per group')

    args = arg_parser.parse_args()
    return args


def main():

    args = get_args()

    os.makedirs(args.tmp_dir, exist_ok=True)

    # check whether key files exist
    if not os.path.exists(args.bin_path):
        print('[ERROR]', 'Binary Path Does Not Exist}')
        exit(0)
    if not os.path.exists(args.ghidra_path):
        print('[ERROR]', 'Ghidra Does Not Exist}')
        exit(0)
    if not os.path.exists(args.script_path):
        print('[ERROR]', 'Script Path Does Not Exist}')
        exit(0)
    if not os.path.exists(os.path.join(args.script_path, args.script_name)):
        print('[ERROR]', 'Script File Does Not Exist}')
        exit(0)

    # get all bins
    bins = []
    for bin_dir in os.listdir(args.bin_path):
        for file in os.listdir(os.path.join(args.bin_path, bin_dir)):
            filepath = os.path.join(args.bin_path, bin_dir, file)
            if is_elf_file(filepath):
                bins.append(filepath)

    print('[INFO]', f'# of bins: {len(bins)}')

    # dispatch task to each core
    group_count = math.ceil(len(bins) / args.workload)

    batch_process_partial = functools.partial(batch_process, args=args)
    with Pool(processes=args.core) as pool:
        for _ in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(bins, args.workload)),
                total=group_count):
            pass

    print('[INFO]', 'Ghidra Script Running Completed')
    shutil.rmtree(args.tmp_dir)


if __name__ == '__main__':
    main()

