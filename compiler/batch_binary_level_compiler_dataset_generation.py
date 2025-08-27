import os
import math
import argparse
import functools

from tqdm import tqdm
from multiprocessing import Pool
from utils import pickle_load, pickle_dump, generator, add_item_to_list_of_dict, Instruction, InternalMethod


def get_args():
    arg_parser = argparse.ArgumentParser(
        description='This script is to load pickles and generate binary-level compiler identification dataset.')
    arg_parser.add_argument('--dataset_path', type=str, default='/dev/shm/split_dataset/train',
                            help='The original dataset')
    arg_parser.add_argument('--output_path',
                            type=str, default='/mnt/ssd1/anonymous/binary_level_compiler_dataset/normalized_instruction/binaries_train.pkl')

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

    instruction_sequence = []

    for func in funcs:
        insts: list[Instruction] = func.get_instructions()

        # `opcode,oper1,oper2,...
        instruction_sequence.extend([inst.get_opcode() if len(
            inst.get_operands()) == 0 else f'{inst.get_opcode()},{",".join(inst.get_operands())}' for inst in insts])

    return tuple(instruction_sequence)


def batch_process(bin_dirs, args) -> dict[str, dict[str, list[tuple]]]:
    # {'compiler1': {'O0': [(inst1, inst2, ...)]}, 'compiler2': {'O0': [(inst1, inst2, ...)]}, ...}
    compiler_opt_dict: dict[str, dict[str, list[tuple]]] = {}

    for bin_dir in bin_dirs:
        for file in os.listdir(os.path.join(args.dataset_path, bin_dir)):
            if file.endswith('.norm_code'):
                compiler = file.split('_')[0]
                opt = file[:-len('.norm_code')].split('_')[-1]
                inst_seq: tuple[str, ...] = get_inst_seq_from_pickle(os.path.join(args.dataset_path, bin_dir, file))
                if compiler not in compiler_opt_dict.keys():
                    compiler_opt_dict[compiler] = {}

                add_item_to_list_of_dict(d=compiler_opt_dict[compiler], k=opt, i=inst_seq)

    return compiler_opt_dict


def main():

    args = get_args()

    compiler_opt_dict = {}

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
        for sub_compiler_opt_dict in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(bin_dirs, args.workload)),
                total=group_count):
            sub_compiler_opt_dict: dict[str, dict[str, list[tuple]]]
            for compiler, opt_dict in sub_compiler_opt_dict.items():
                if compiler not in compiler_opt_dict.keys():
                    compiler_opt_dict[compiler] = opt_dict
                else:
                    for opt, funcs in opt_dict.items():
                        if opt not in compiler_opt_dict[compiler].keys():
                            compiler_opt_dict[compiler][opt] = funcs
                        else:
                            compiler_opt_dict[compiler][opt].extend(funcs)

    print('[INFO]', 'Dataset Generation Completed')
    pickle_dump(compiler_opt_dict, args.output_path)


if __name__ == '__main__':
    main()
