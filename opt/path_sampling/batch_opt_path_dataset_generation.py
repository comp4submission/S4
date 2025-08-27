import os
import math
import glob
import random
import argparse
import functools
import setproctitle

from tqdm import tqdm
from multiprocessing import Pool
from collections import namedtuple
from utils import (pickle_dump, generator, add_item_to_list_of_dict, extend_items_to_list_of_dict,
                   sample_instructions_from_fcg, Instruction, InternalMethod)


def get_args():
    arg_parser = argparse.ArgumentParser(
        description='This script is to generate byte-based binary-level optimization identification dataset.')
    arg_parser.add_argument('--dataset_path', type=str, default='/dev/shm/split_dataset/train',
                            help='The original dataset')
    arg_parser.add_argument('--output_path', type=str, default='/dev/shm/opt/sample_inst/binaries_train.pkl')

    arg_parser.add_argument('--inst_threshold', type=int, default=40000, help='# of sampled instructions')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count() // 2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=1, help='workload per group')

    args = arg_parser.parse_args()
    return args


def batch_process(pkl_files, args):
    # {'O0': [(inst1, inst2, ...), (inst1, inst2, ...)], 'O1': [(inst1, inst2, ...), (inst1, inst2, ...)], ...}
    binaries: dict[str, list[tuple]] = {}

    for filepath in pkl_files:

        file = os.path.basename(filepath)

        if file.endswith('.fcg'):
            compiler, opt = file[:-len('.fcg')].split('_')

            setproctitle.setproctitle(f'Extracting: {filepath}')

            insts: list[Instruction] = sample_instructions_from_fcg(filepath, threshold=args.inst_threshold)
            inst_seq = tuple([inst.get_opcode() if len(inst.get_operands()) == 0 else f'{inst.get_opcode()},{",".join(inst.get_operands())}' for inst in insts])
            add_item_to_list_of_dict(d=binaries, k=opt, i=inst_seq)

    return binaries


def main(args=None):

    if args is None:
        args = get_args()

    binaries: dict[str, list[tuple]] = {}

    # check whether the dataset path exist
    if not os.path.exists(args.dataset_path):
        print('[ERROR]', 'Dataset Path Not Exist}')
        exit(0)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    # get all pkl files
    pkl_files = glob.glob(os.path.join(args.dataset_path, '**', '*.fcg'), recursive=True)
    random.shuffle(pkl_files)

    # dispatch task to each core
    group_count = math.ceil(len(pkl_files) / args.workload)

    batch_process_partial = functools.partial(batch_process, args=args)

    with Pool(processes=args.core) as pool:
        for sub_binaries in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(pkl_files, args.workload)),
                total=group_count):
            sub_binaries: dict[str, list[tuple]]
            for opt, inst_seqs in sub_binaries.items():
                extend_items_to_list_of_dict(d=binaries, k=opt, items=inst_seqs)

    print('[INFO]', 'Dataset Generation Completed')
    pickle_dump(binaries, args.output_path)


if __name__ == '__main__':
    Args = namedtuple('Args', ['dataset_path', 'output_path', 'inst_threshold', 'core', 'workload'])
    main(Args(dataset_path='/dev/shm/split_dataset/train',
              output_path='/dev/shm/opt/sample_inst_adapt/binaries_train.pkl',
              inst_threshold=40000, core=int(os.cpu_count() * 0.9), workload=10))
    main(Args(dataset_path='/dev/shm/split_dataset/test',
              output_path='/dev/shm/opt/sample_inst_adapt/binaries_test.pkl',
              inst_threshold=40000, core=int(os.cpu_count() * 0.9), workload=10))