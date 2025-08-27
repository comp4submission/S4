import os
import math
import json
import argparse
import functools

from tqdm import tqdm
from multiprocessing import Pool
from utils import pickle_dump, generator, add_item_to_list_of_dict, extend_items_to_list_of_dict


def get_args():
    arg_parser = argparse.ArgumentParser(
        description='This script is to generate pcode-based binary-level optimization identification dataset.')
    arg_parser.add_argument('--dataset_path', type=str, default='/dev/shm/split_dataset/train',
                            help='The original dataset')
    arg_parser.add_argument('--output_path', type=str,
                            default='/mnt/ssd1/anonymous/binary_level_opt_dataset/pcode/binaries_train.pkl')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count() // 2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=5, help='workload per group')

    args = arg_parser.parse_args()
    return args


def get_pcode_seq_from_json(json_path) -> tuple[str, ...]:
    with open(json_path) as f:
        addr_pcode_map: dict[str, list[str, ...]] = json.load(f)

    # sort the functions by address
    # key: '0004ba' value: ['', '']
    ordered_funcs = sorted(addr_pcode_map.items(), key=lambda kv: int(kv[0], 16))
    pcode_list = []

    for item in ordered_funcs:
        pcode_list.extend(item[1])

    return tuple(pcode_list)


def batch_process(bin_dirs, args):
    # {'O0': [(inst1, inst2, ...), (inst1, inst2, ...)], 'O1': [(inst1, inst2, ...), (inst1, inst2, ...)], ...}
    binaries: dict[str, list[tuple]] = {}

    for bin_dir in bin_dirs:
        for file in os.listdir(os.path.join(args.dataset_path, bin_dir)):
            if file.endswith('.pcode.json'):
                opt = file[:-len('.pcode.json')].split('_')[-1]
                pcode_seq = get_pcode_seq_from_json(os.path.join(args.dataset_path, bin_dir, file))

                add_item_to_list_of_dict(d=binaries, k=opt, i=pcode_seq)

    return binaries


def main():

    args = get_args()

    binaries: dict[str, list[tuple]] = {}

    # check whether the dataset path exist
    if not os.path.exists(args.dataset_path):
        print('[ERROR]', 'Dataset Path Not Exist}')
        exit(0)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # get all bin_dirs
    bin_dirs = os.listdir(args.dataset_path)

    # dispatch task to each core
    group_count = math.ceil(len(bin_dirs) / args.workload)

    batch_process_partial = functools.partial(batch_process, args=args)

    with Pool(processes=args.core) as pool:
        for sub_binaries in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(bin_dirs, args.workload)),
                total=group_count):
            sub_binaries: dict[str, list[tuple]]
            for opt, inst_seqs in sub_binaries.items():
                extend_items_to_list_of_dict(d=binaries, k=opt, items=inst_seqs)

    print('[INFO]', 'Dataset Generation Completed')
    pickle_dump(binaries, args.output_path)


if __name__ == '__main__':
    main()
