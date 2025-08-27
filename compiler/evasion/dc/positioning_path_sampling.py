import os
import glob
import math
import json
import joblib
import random
import functools
import networkx as nx

from tqdm import tqdm
from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from utils import InternalMethod, Instruction, pickle_load, generator
from sklearn.feature_extraction.text import CountVectorizer


def sample_element_by_order(l):
    """
    Sample an element from a list based on the order of each element
    The selection probability is proportional to the order of the element.
    The earlier the element, the easier it is to be selected.
    The selected element will be put at the end of the list.
    """
    temp_index = []
    for i in range(len(l)):
        temp_index.extend([i] * (len(l) - i))

    selected_index = random.choice(temp_index)

    return l[:selected_index] + l[selected_index+1:] + l[selected_index:selected_index+1]


def sample_instructions_from_fcg(fcg: nx.DiGraph, threshold=10000) -> list[Instruction]:

    n_dc_map = nx.degree_centrality(fcg)

    # sort nodes by degree centrality from low to high
    sorted_nodes = [n for n, dc in sorted(n_dc_map.items(), key=lambda x: x[1])]

    # get the total number of instructions
    total_instructions = sum([len(fcg.nodes[n]['data'].get_instructions()) for n in sorted_nodes])
    if threshold == 0:
        threshold = int(total_instructions * 0.7)

    sampled_instruction_list: list[Instruction] = []

    while True:
        sorted_nodes = sample_element_by_order(sorted_nodes)
        sampled_node = sorted_nodes[-1]

        sampled_instruction_list.extend(explore(fcg, sampled_node))

        if len(sampled_instruction_list) > threshold:
            return sampled_instruction_list


def explore(g: nx.DiGraph, start_node: int) -> list[Instruction]:
    paths: list[int] = []
    instruction_list: list[Instruction] = []

    def get_successors(n):
        _successors = list(g.successors(n))

        for old_n in paths:
            if old_n in _successors:
                _successors.remove(old_n)

        return _successors

    # start explore
    cnt_node = start_node
    m: InternalMethod = g.nodes[cnt_node]['data']
    instruction_list.extend(m.get_instructions())
    paths.append(cnt_node)

    while True:
        successors = get_successors(cnt_node)
        if len(successors) == 0:
            break

        cnt_node = random.choice(successors)
        m: InternalMethod = g.nodes[cnt_node]['data']
        instruction_list.extend(m.get_instructions())
        paths.append(cnt_node)

    return instruction_list


def get_inst_object_list_from_pickle(fcg: nx.DiGraph, max_inst_seq_length) -> list[Instruction]:
    funcs = nx.get_node_attributes(fcg, 'data').values()
    funcs = sorted(funcs, key=lambda x: x.get_start_addr())

    instruction_obj_list = []
    for func in funcs:
        insts: list[Instruction] = func.get_instructions()
        instruction_obj_list.extend(insts)

    return instruction_obj_list[:min(max_inst_seq_length, len(instruction_obj_list))]


def calculate_covering_cost(signature: tuple, idx_weight_map: dict[int, int]) -> float:
    # the idx list of byte of interest
    byte_of_interest = list(range(*signature))

    mean = sum(byte_of_interest) / len(byte_of_interest)

    covering_cost = 0
    for index in byte_of_interest:
        covering_cost += idx_weight_map[index] * ((index - mean)**2)

    return covering_cost


def transform_inst_obj_list_to_inst_seq(inst_obj_list: list[Instruction]) -> tuple[str, ...]:
    return tuple([inst.get_opcode() if len(inst.get_operands()) == 0
                  else f'{inst.get_opcode()},{",".join(inst.get_operands())}' for inst in inst_obj_list])


def scan_by_inst_tuple(inst_tuple, model, vectorizer) -> int:
    corpus = [' '.join(inst_tuple)]

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


def get_opcodeweight_from_json(json_path):

    with open(json_path) as f:
        opcodeweight_map = json.load(f)

    return opcodeweight_map


def positioning(pkl_path, model, vectorizer, max_inst_seq_length):
    global byte_constraint

    fcg = pickle_load(pkl_path)

    original_inst_obj_list = get_inst_object_list_from_pickle(fcg, max_inst_seq_length)

    opcodeweight_path = f'{pkl_path[:-len(".fcg")]}.opcodeweight.json'

    opcodeweight: dict[str, int] = get_opcodeweight_from_json(opcodeweight_path)

    # get the initial scan result
    sampled_insts = sample_instructions_from_fcg(fcg, threshold=max_inst_seq_length)
    inst_tuple = transform_inst_obj_list_to_inst_seq(sampled_insts)
    original_scan_result = scan_by_inst_tuple(inst_tuple, model, vectorizer)

    signatures = []
    worklist: list[tuple] = [(0, len(original_inst_obj_list))]

    while len(worklist) > 0:
        cnt_interval = worklist.pop()

        black_addr_list = [original_inst_obj_list[i].get_address() for i in range(cnt_interval[0], cnt_interval[1])]

        sampled_insts = sample_instructions_from_fcg(fcg, threshold=max_inst_seq_length)
        # replace the blacklisted instructions with nops
        for i in range(len(sampled_insts)):
            if sampled_insts[i].get_address() in black_addr_list:
                sampled_insts[i] = Instruction(address=sampled_insts[i].get_address(),
                                               size=sampled_insts[i].get_size(), opcode='nop', operands=[], regs=[])
        inst_tuple = transform_inst_obj_list_to_inst_seq(sampled_insts)
        cnt_result = scan_by_inst_tuple(inst_tuple, model, vectorizer)
        # check if the scan result changes
        if cnt_result != original_scan_result:
            # the cnt_interval contains signature
            # print('Signature in the interval: ', cnt_interval)

            involved_byte_num = 0
            for i in range(cnt_interval[0], cnt_interval[1]):
                involved_byte_num += original_inst_obj_list[i].get_size()

            if involved_byte_num <= byte_constraint:
                signatures.append(cnt_interval)
            else:
                # the cnt_interval is too long, split it
                worklist.extend(split(cnt_interval))

    # # validation
    # validated_sig = []
    # for signature in signatures:
    #     is_robust = True
    #     # validate 5 times
    #     for _ in range(5):
    #         black_addr_list = [original_inst_obj_list[i].get_address() for i in range(signature[0], signature[1])]
    #         sampled_insts = sample_instructions_from_fcg(fcg, threshold=max_inst_seq_length)
    #         # replace the blacklisted instructions with nops
    #         for i in range(len(sampled_insts)):
    #             if sampled_insts[i].get_address() in black_addr_list:
    #                 sampled_insts[i] = Instruction(address=sampled_insts[i].get_address(),
    #                                                size=sampled_insts[i].get_size(), opcode='nop', operands=[], regs=[])
    #         inst_tuple = transform_inst_obj_list_to_inst_seq(sampled_insts)
    #         cnt_result = scan_by_inst_tuple(inst_tuple, model, vectorizer)
    #         if cnt_result == original_scan_result:
    #             is_robust = False
    #             break
    #
    #     if is_robust:
    #         validated_sig.append(signature)
    #
    # print('Filtered out {} non-robust signatures'.format(len(signatures) - len(validated_sig)))
    # signatures = validated_sig

    if len(signatures) == 0:
        print(f'Evasion Failed under Constraint={byte_constraint}')
        return None

    # refine the signatures and find the one with minimum covering cost
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

            black_addr_list = [original_inst_obj_list[i].get_address() for i in
                               range(cnt_interval[0], cnt_interval[1])]

            sampled_insts = sample_instructions_from_fcg(fcg, threshold=max_inst_seq_length)
            # replace the blacklisted instructions with nops
            for i in range(len(sampled_insts)):
                if sampled_insts[i].get_address() in black_addr_list:
                    sampled_insts[i] = Instruction(address=sampled_insts[i].get_address(),
                                                   size=sampled_insts[i].get_size(), opcode='nop', operands=[],
                                                   regs=[])
            inst_tuple = transform_inst_obj_list_to_inst_seq(sampled_insts)
            cnt_result = scan_by_inst_tuple(inst_tuple, model, vectorizer)

            if cnt_result != original_scan_result:
                #####
                idx_weight_map = {}
                total_bytes = 0
                for inst_obj in original_inst_obj_list[cnt_interval[0]:cnt_interval[1]]:
                    opcode = inst_obj.get_opcode().lower()
                    for _ in range(inst_obj.get_size()):
                        idx_weight_map[total_bytes] = opcodeweight[opcode] if opcode in opcodeweight else 1
                        total_bytes += 1
                cnt_covering_cost = calculate_covering_cost((0, total_bytes), idx_weight_map)
                #####

                if cnt_covering_cost < min_covering_cost:
                    min_covering_cost = cnt_covering_cost

                worklist.extend(split(cnt_interval))

    return min_covering_cost


def batch_process(pkl_paths, args):
    model_file = args['model_file']
    vocab_path = args['vocab_path']
    max_inst_seq_length = args['max_inst_seq_length']

    model = joblib.load(model_file)
    vectiorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None,
                                  vocabulary=joblib.load(vocab_path))

    min_covering_costs = []
    successful_cnt = 0
    failed_cnt = 0

    for pkl_path in pkl_paths:

        min_covering_cost = positioning(pkl_path=pkl_path, model=model, vectorizer=vectiorizer, max_inst_seq_length=max_inst_seq_length)

        if min_covering_cost is None:
            failed_cnt += 1
            continue

        #  successful evasion
        successful_cnt += 1
        min_covering_costs.append(min_covering_cost)

    return min_covering_costs, successful_cnt, failed_cnt


def main():

    TESTSET_PATH = '/dev/shm/split_dataset/test'
    vocab_path = '/mnt/ssd1/anonymous/binary_level_compiler_dataset/sample_inst/vectorizer_vocab.dat'
    model_file = '/mnt/ssd1/anonymous/binary_level_compiler_dataset/sample_inst/model_40000.dat'
    max_inst_seq_length = int(os.path.basename(model_file).split('.')[0].split('_')[-1])

    args = {
        'vocab_path': vocab_path,
        'model_file': model_file,
        'max_inst_seq_length': max_inst_seq_length
    }

    pkl_paths = glob.glob(os.path.join(TESTSET_PATH, '**', '*.fcg'), recursive=True)

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
