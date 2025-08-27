import os
import math
import argparse
import functools
import setproctitle

from tqdm import tqdm
from multiprocessing import Pool
from utils import pickle_load, pickle_dump, generator, Instruction, InternalMethod, add_item_to_list_of_dict, extend_items_to_list_of_dict


def get_args():
    arg_parser = argparse.ArgumentParser(
        description='This script is to load pickles and generate binary-level optimization identification dataset.')
    arg_parser.add_argument('--dataset_path', type=str, default='/dev/shm/split_dataset/train',
                            help='The original dataset')
    arg_parser.add_argument('--output_path', type=str, default='/dev/shm/DIComP/opt/binaries_train.pkl')

    arg_parser.add_argument('--core', type=int, default=os.cpu_count() // 2, help='cores involved')
    arg_parser.add_argument('--workload', type=int, default=1, help='workload per group')

    args = arg_parser.parse_args()
    return args


def get_inst_seq_from_pickle(pkl_path: str):
    if not os.path.exists(pkl_path):
        return None

    name_func_map: dict[str, InternalMethod] = pickle_load(pkl_path)
    funcs: list[InternalMethod] = list(name_func_map.values())
    # sort functions by address
    funcs.sort(key=lambda m: m.get_start_addr())

    mnemonics_seq: list[str] = []
    reg_seq: list[str] = []
    func_length_seq: list[str] = []

    for func in funcs:
        insts: list[Instruction] = func.get_instructions()

        mnemonics_seq.extend([inst.get_opcode() for inst in insts])

        for inst in insts:
            reg_seq.extend(inst.get_regs())

        func_length_seq.append(str(len(insts)))

    return tuple(mnemonics_seq + reg_seq + func_length_seq)


def batch_process(bin_dirs, args):
    # {'O0': [(inst1, inst2, ...), (inst1, inst2, ...)], 'O1': [(inst1, inst2, ...), (inst1, inst2, ...)], ...}
    binaries: dict[str, list[tuple]] = {}

    for bin_dir in bin_dirs:
        for file in os.listdir(os.path.join(args.dataset_path, bin_dir)):
            if file.endswith('.norm_code'):
                opt = file[:-len('.norm_code')].split('_')[-1]

                setproctitle.setproctitle(f'Extracting: {os.path.join(args.dataset_path, bin_dir, file)}')

                inst_seq: tuple[str, ...] = get_inst_seq_from_pickle(os.path.join(args.dataset_path, bin_dir, file))
                add_item_to_list_of_dict(d=binaries, k=opt, i=inst_seq)

    return binaries


def main():

    args = get_args()

    binaries: dict[str, list[tuple]] = {}

    # check whether the dataset path exist
    if not os.path.exists(args.dataset_path):
        print('[ERROR]', 'Dataset Path Not Exist}')
        exit(0)

    if not os.path.exists(os.path.dirname(args.output_path)):
        os.makedirs(os.path.dirname(args.output_path))

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
