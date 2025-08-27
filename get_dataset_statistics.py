import os
import glob
import joblib

from tqdm import tqdm
from utils import InternalMethod, Instruction


def main():

    DATASET_PATH = '/dev/shm/split_dataset'
    pkl_paths = glob.glob(os.path.join(DATASET_PATH, '**', '*.norm_code'), recursive=True)

    inst_count_list = []
    func_count_list = []

    for pkl_path in tqdm(pkl_paths):
        name_func_map: dict[str, InternalMethod] = joblib.load(pkl_path)
        funcs: list[InternalMethod] = list(name_func_map.values())
        # sort functions by address
        funcs.sort(key=lambda m: m.get_start_addr())

        inst_count = 0
        for func in funcs:
            insts: list[Instruction] = func.get_instructions()
            inst_count += len(insts)

        inst_count_list.append(inst_count)
        func_count_list.append(len(funcs))

    print('Total Files:', len(pkl_paths))
    print('Min, Max, Avg (Instruction):', min(inst_count_list), max(inst_count_list), sum(inst_count_list) / len(inst_count_list))
    print('Min, Max, Avg (Function):', min(func_count_list), max(func_count_list), sum(func_count_list) / len(func_count_list))


if __name__ == '__main__':
    main()
