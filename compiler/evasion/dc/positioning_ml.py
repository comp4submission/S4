import os
import glob
import math
import json
import joblib
import random
import functools

from tqdm import tqdm
from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from utils import InternalMethod, Instruction, pickle_load, generator


def get_inst_object_list_from_pickle(pkl_path: str, max_inst_seq_length) -> list[Instruction]:

    name_func_map: dict[str, InternalMethod] = pickle_load(pkl_path)
    funcs: list[InternalMethod] = list(name_func_map.values())
    # sort functions by address
    funcs.sort(key=lambda m: m.get_start_addr())

    instruction_obj_list = []
    for func in funcs:
        insts: list[Instruction] = func.get_instructions()
        instruction_obj_list.extend(insts)

    return instruction_obj_list[:min(max_inst_seq_length, len(instruction_obj_list))]


def transform_inst_obj_list_to_inst_seq(inst_obj_list: list[Instruction]) -> tuple[str, ...]:
    return tuple([inst.get_opcode() if len(inst.get_operands()) == 0 else f'{inst.get_opcode()},{",".join(inst.get_operands())}' for inst in inst_obj_list])


def calculate_covering_cost(signature: tuple, idx_weight_map: dict[int, int]) -> float:
    # the idx list of byte of interest
    byte_of_interest = list(range(*signature))

    mean = sum(byte_of_interest) / len(byte_of_interest)

    covering_cost = 0
    for index in byte_of_interest:
        covering_cost += idx_weight_map[index] * ((index - mean)**2)

    return covering_cost


def scan_by_inst_obj_list(inst_obj_list, model, vectorizer) -> int:
    corpus = []
    inst_tuple = transform_inst_obj_list_to_inst_seq(inst_obj_list)

    corpus.append(' '.join(inst_tuple))

    X_test = vectorizer.transform(corpus).toarray()
    preds = model.predict(X_test)

    return preds[0]


def split(original_interval: tuple) -> list[tuple]:
    """
    Split the interval in two parts.
    """
    start, end = original_interval
    middle = (start + end) // 2

    if start == middle or end == middle:
        return []

    return [(start, middle), (middle, end)]


def patch(original_inst_obj_list: list[Instruction], interval: tuple) -> list[Instruction]:
    """
    Patch the binary with the new content.
    """
    start, end = interval

    patched_part = []
    for i in range(start, end):
        addr = original_inst_obj_list[i].get_address()
        size = original_inst_obj_list[i].get_size()
        patched_part.append(Instruction(address=addr, size=size, opcode='nop', operands=[], regs=[]))
        # for inc in range(size):
        #     patched_part.append(Instruction(address=addr + inc, size=1, opcode='nop', operands=[], regs=[]))

    new_inst_obj_list = original_inst_obj_list[:start] + patched_part + original_inst_obj_list[end:]

    assert len(original_inst_obj_list) == len(new_inst_obj_list)

    return new_inst_obj_list


def get_opcodeweight_from_json(json_path):

    with open(json_path) as f:
        opcodeweight_map = json.load(f)

    return opcodeweight_map


def positioning(pkl_path, model, vectorizer, max_inst_seq_length):

    global  byte_constraint

    original_inst_obj_list: list[Instruction] = get_inst_object_list_from_pickle(pkl_path, max_inst_seq_length)

    opcodeweight_path = f'{pkl_path[:-len(".norm_code")]}.opcodeweight.json'

    opcodeweight: dict[str, int] = get_opcodeweight_from_json(opcodeweight_path)

    original_scan_result = scan_by_inst_obj_list(original_inst_obj_list, model, vectorizer)

    signatures = []
    worklist: list[tuple] = [(0, len(original_inst_obj_list))]

    while len(worklist) > 0:
        cnt_interval = worklist.pop()

        # check if the scan result changes
        new_inst_obj_list: list[Instruction] = patch(original_inst_obj_list, cnt_interval)
        cnt_result = scan_by_inst_obj_list(new_inst_obj_list, model, vectorizer)

        if cnt_result != original_scan_result:
            # the cnt_interval contains signature
            # print('Signature in the interval: ', cnt_interval)

            involved_byte_num = 0
            for i in range(cnt_interval[0], cnt_interval[1]):
                involved_byte_num += original_inst_obj_list[i].get_size()

            if involved_byte_num <= byte_constraint:
                signatures.append(cnt_interval)
            else:
                worklist.extend(split(cnt_interval))

    if len(signatures) == 0:
        print(f'Evasion Failed under Constraint={byte_constraint}')
        return None

    # refine the signatures and find the one with minimum covering cost

    # calculate the covering cost of the first interval as initial value
    #####
    idx_weight_map = {}
    total_bytes = 0
    for inst_obj in original_inst_obj_list[signatures[0][0]:signatures[0][1]]:
        opcode = inst_obj.get_opcode().lower()
        for _ in range(inst_obj.get_size()):
            idx_weight_map[total_bytes] = opcodeweight[opcode] if opcode in opcodeweight else 1
            total_bytes += 1
    min_covering_cost = calculate_covering_cost((0, total_bytes), idx_weight_map)
    #####

    for sig in signatures:
        worklist = split(sig)
        while len(worklist) > 0:
            cnt_interval = worklist.pop()
            new_inst_obj_list: list[Instruction] = patch(original_inst_obj_list, cnt_interval)
            cnt_result = scan_by_inst_obj_list(new_inst_obj_list, model, vectorizer)

            if cnt_result != original_scan_result:
                #######
                idx_weight_map = {}
                total_bytes = 0
                for inst_obj in original_inst_obj_list[cnt_interval[0]:cnt_interval[1]]:
                    opcode = inst_obj.get_opcode().lower()
                    for _ in range(inst_obj.get_size()):
                        idx_weight_map[total_bytes] = opcodeweight[opcode] if opcode in opcodeweight else 1
                        total_bytes += 1
                cnt_covering_cost = calculate_covering_cost((0, total_bytes), idx_weight_map)
                #######

                if cnt_covering_cost < min_covering_cost:
                    min_covering_cost = cnt_covering_cost

                worklist.extend(split(cnt_interval))

    return min_covering_cost


def batch_process(pkl_paths, args):
    model_file = args['model_file']
    vocab_path = args['vocab_path']
    max_inst_seq_length = args['max_inst_seq_length']

    model = joblib.load(model_file)
    vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None,
                                  vocabulary=joblib.load(vocab_path))

    min_covering_costs = []
    successful_cnt = 0
    failed_cnt = 0

    for pkl_path in pkl_paths:

        min_covering_cost = positioning(pkl_path=pkl_path, model=model, vectorizer=vectorizer, max_inst_seq_length=max_inst_seq_length)

        if min_covering_cost is None:
            failed_cnt += 1
            continue

        #  successful evasion
        successful_cnt += 1
        min_covering_costs.append(min_covering_cost)

    return min_covering_costs, successful_cnt, failed_cnt


def main():

    TESTSET_PATH = '/dev/shm/split_dataset/test'
    vocab_path = '/mnt/ssd1/anonymous/binary_level_compiler_dataset/normalized_instruction/vectorizer_vocab.dat'
    model_file = '/mnt/ssd1/anonymous/binary_level_compiler_dataset/normalized_instruction/model_32000.dat'
    max_inst_seq_length = int(os.path.basename(model_file).split('.')[0].split('_')[-1])

    args = {
        'vocab_path': vocab_path,
        'model_file': model_file,
        'max_inst_seq_length': max_inst_seq_length
    }

    pkl_paths = glob.glob(os.path.join(TESTSET_PATH, '**', '*.norm_code'), recursive=True)

    workload = 5
    group_count = math.ceil(len(pkl_paths) / workload)

    batch_process_partial = functools.partial(batch_process, args=args)

    min_covering_costs = []
    successful_cnt = 0
    failed_cnt = 0

    with Pool(processes=os.cpu_count() // 4) as pool:
        for sub_min_covering_costs, sub_successful_cnt, sub_failed_cnt in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(pkl_paths, workload)),
                total=group_count):

            min_covering_costs.extend(sub_min_covering_costs)
            successful_cnt += sub_successful_cnt
            failed_cnt += sub_failed_cnt

            if successful_cnt > 0:
                print('Minimum Covering Cost: ', format(sum(min_covering_costs) / len(min_covering_costs), '.4f'))
            print('Success Rate: ', format(successful_cnt / (successful_cnt + failed_cnt), '.4f'))

    print('Minimum Covering Cost: ', format(sum(min_covering_costs) / len(min_covering_costs), '.4f'))
    print('Success Rate: ', format(successful_cnt / (successful_cnt + failed_cnt), '.4f'))


if __name__ == '__main__':
    byte_constraint = 20
    main()
