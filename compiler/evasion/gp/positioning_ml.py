import os
import glob
import math
import copy
import json
import pygad
import joblib
import random
import functools
import numpy as np

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


def calculate_covering_cost_by_boi(byte_of_interest, idx_weight_map: dict[int, int]) -> float:

    mean = sum(byte_of_interest) / len(byte_of_interest)

    covering_cost = 0
    for index in byte_of_interest:
        covering_cost += idx_weight_map[index] * ((index - mean)**2)

    return covering_cost


def transform_inst_obj_list_to_inst_seq(inst_obj_list: list[Instruction]) -> tuple[str, ...]:
    return tuple([inst.get_opcode() if len(inst.get_operands()) == 0 else f'{inst.get_opcode()},{",".join(inst.get_operands())}' for inst in inst_obj_list])


def scan_by_inst_obj_list(inst_obj_list) -> int:
    global model, vectorizer

    corpus = []
    inst_tuple = transform_inst_obj_list_to_inst_seq(inst_obj_list)

    corpus.append(' '.join(inst_tuple))

    X_test = vectorizer.transform(corpus).toarray()
    preds = model.predict(X_test)

    return preds[0]


def get_opcodeweight_from_json(json_path):

    with open(json_path) as f:
        opcodeweight_map = json.load(f)

    return opcodeweight_map


def is_evade(perturbation):
    global original_scan_result, original_inst_obj_list

    new_inst_obj_list = copy.deepcopy(original_inst_obj_list)

    for i in range(len(perturbation)):
        if perturbation[i] == 1:
            new_inst_obj_list[i] = Instruction(address=original_inst_obj_list[i].get_address(), size=original_inst_obj_list[i].get_size(), opcode='nop', operands=[], regs=[])

    new_scan_result = scan_by_inst_obj_list(new_inst_obj_list)

    # scan result changed
    # True: evade
    # False: not evade
    return new_scan_result != original_scan_result


def fitness_func(ga_instance, solution, solution_idx):
    fitness = is_evade(solution) + (1.0 / (sum(solution) + 2))
    return fitness


def callback_generation(ga_instance):
    print(f"Changed instructions: {sum(ga_instance.best_solution()[0])}")
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")


def positioning(pkl_path, max_inst_seq_length):

    global original_inst_obj_list, original_scan_result

    original_inst_obj_list = get_inst_object_list_from_pickle(pkl_path, max_inst_seq_length)

    num_generations = 10000  # Number of generations.
    num_parents_mating = 5  # Number of solutions to be selected as parents in the mating pool.
    sol_per_pop = 10  # Number of solutions in the population.
    num_genes = len(original_inst_obj_list)

    opcodeweight_path = f'{pkl_path[:-len(".norm_code")]}.opcodeweight.json'
    opcodeweight: dict[str, int] = get_opcodeweight_from_json(opcodeweight_path)

    original_scan_result = scan_by_inst_obj_list(original_inst_obj_list)

    initial_population = np.zeros(shape=(sol_per_pop, num_genes), dtype=int)

    # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           initial_population=initial_population,
                           mutation_percent_genes=1,
                           on_generation=callback_generation,
                           gene_space=[0, 1],
                           gene_type=int,
                           save_solutions=True,
                           stop_criteria=['saturate_20', 'reach_1'])
    ga_instance.run()

    best_solution, best_solution_fitness, best_solution_idx = ga_instance.best_solution()
    print('best_solution_fitness: ', best_solution_fitness)

    if not is_evade(best_solution):
        return None, None

    # get the corresponding index of the adversarial solution
    adversarial_solution_idx = np.where(best_solution == 1)[0]

    print('new num_genes: ', len(adversarial_solution_idx))

    def fitness_func_2(ga_instance, solution, solution_idx):
        # initial the modification
        modification = [0 for _ in range(len(original_inst_obj_list))]

        for i in range(len(solution)):
            if solution[i] == 1:
                modification[adversarial_solution_idx[i]] = 1

        # try to minimize the number of changed bytes while keeping the binary evading the rules
        fitness = is_evade(modification) + 1.0 / (sum(modification) + 1)
        return fitness

    ga_instance2 = pygad.GA(num_generations=num_generations,
                            num_parents_mating=num_parents_mating,
                            fitness_func=fitness_func_2,
                            sol_per_pop=sol_per_pop,
                            num_genes=len(adversarial_solution_idx),
                            on_generation=callback_generation,
                            gene_space=[0, 1],
                            mutation_percent_genes=20,
                            gene_type=int,
                            stop_criteria='saturate_10',
                            save_solutions=True)
    ga_instance2.run()

    best_partial_solution = ga_instance2.best_solution()[0]

    print(f"Changed instructions: {sum(best_partial_solution)}")

    adv_modification = [0 for _ in range(len(original_inst_obj_list))]

    for i in range(len(best_partial_solution)):
        if best_partial_solution[i] == 1:
            adv_modification[adversarial_solution_idx[i]] = 1

    adv_modification = np.array(adv_modification)
    ioi = np.where(adv_modification == 1)[0]

    # now we have the index of the instruction to be modified
    # we need to calculate the number of changed bytes
    boi = []
    idx_weight_map = {}

    for inst_idx in ioi:
        inst: Instruction = original_inst_obj_list[inst_idx]
        for i in range(inst.get_address(), inst.get_address() + inst.get_size()):
            boi.append(i)
            opcode = inst.get_opcode().lower()
            idx_weight_map[i] = opcodeweight[opcode] if opcode in opcodeweight else 1

    covering_cost = calculate_covering_cost_by_boi(boi, idx_weight_map)


    return len(boi), covering_cost


def batch_process(pkl_paths, args):

    global model, vectorizer

    model_file = args['model_file']
    vocab_path = args['vocab_path']
    max_inst_seq_length = args['max_inst_seq_length']

    model = joblib.load(model_file)
    vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None,
                                  vocabulary=joblib.load(vocab_path))

    covering_costs_10 = []
    covering_costs_20 = []
    covering_costs_30 = []

    successful_cnt_10 = 0
    failed_cnt_10 = 0

    successful_cnt_20 = 0
    failed_cnt_20 = 0

    successful_cnt_30 = 0
    failed_cnt_30 = 0

    for pkl_path in pkl_paths:

        num_changed_bytes, covering_cost = positioning(pkl_path=pkl_path, max_inst_seq_length=max_inst_seq_length)

        if covering_cost is None:
            failed_cnt_10 += 1
            failed_cnt_20 += 1
            failed_cnt_30 += 1
            continue

        # successful evasion
        if num_changed_bytes <= 10:

            successful_cnt_10 += 1
            covering_costs_10.append(covering_cost)

            successful_cnt_20 += 1
            covering_costs_20.append(covering_cost)

            successful_cnt_30 += 1
            covering_costs_30.append(covering_cost)

        elif num_changed_bytes <= 20:

            successful_cnt_20 += 1
            covering_costs_20.append(covering_cost)

            successful_cnt_30 += 1
            covering_costs_30.append(covering_cost)

            failed_cnt_10 += 1
        elif num_changed_bytes <= 30:
            successful_cnt_30 += 1
            covering_costs_30.append(covering_cost)
            failed_cnt_10 += 1
            failed_cnt_20 += 1
        else:
            failed_cnt_10 += 1
            failed_cnt_20 += 1
            failed_cnt_30 += 1

    return covering_costs_10, successful_cnt_10, failed_cnt_10, covering_costs_20, successful_cnt_20, failed_cnt_20, covering_costs_30, successful_cnt_30, failed_cnt_30


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

    workload = 3
    group_count = math.ceil(len(pkl_paths) / workload)

    batch_process_partial = functools.partial(batch_process, args=args)

    covering_costs_10 = []
    covering_costs_20 = []
    covering_costs_30 = []

    successful_cnt_10 = 0
    failed_cnt_10 = 0

    successful_cnt_20 = 0
    failed_cnt_20 = 0

    successful_cnt_30 = 0
    failed_cnt_30 = 0

    with Pool(processes=2) as pool:
        for sub_covering_costs_10, sub_successful_cnt_10, sub_failed_cnt_10, sub_covering_costs_20, sub_successful_cnt_20, sub_failed_cnt_20, sub_covering_costs_30, sub_successful_cnt_30, sub_failed_cnt_30 in tqdm(
                pool.imap_unordered(func=batch_process_partial, iterable=generator(pkl_paths, workload)),
                total=group_count):

            covering_costs_10.extend(sub_covering_costs_10)
            covering_costs_20.extend(sub_covering_costs_20)
            covering_costs_30.extend(sub_covering_costs_30)

            successful_cnt_10 += sub_successful_cnt_10
            failed_cnt_10 += sub_failed_cnt_10

            successful_cnt_20 += sub_successful_cnt_20
            failed_cnt_20 += sub_failed_cnt_20

            successful_cnt_30 += sub_successful_cnt_30
            failed_cnt_30 += sub_failed_cnt_30

            if successful_cnt_10 > 0:
                print('Covering Cost (byte=10): ', format(sum(covering_costs_10) / len(covering_costs_10), '.4f'))
                print('Success Rate (byte=10): ',
                      format(successful_cnt_10 / (successful_cnt_10 + failed_cnt_10), '.4f'))
            if successful_cnt_20 > 0:
                print('Covering Cost (byte=20): ', format(sum(covering_costs_20) / len(covering_costs_20), '.4f'))
                print('Success Rate (byte=20): ',
                      format(successful_cnt_20 / (successful_cnt_20 + failed_cnt_20), '.4f'))
            if successful_cnt_30 > 0:
                print('Covering Cost (byte=30): ', format(sum(covering_costs_30) / len(covering_costs_30), '.4f'))
                print('Success Rate (byte=30): ',
                      format(successful_cnt_30 / (successful_cnt_30 + failed_cnt_30), '.4f'))

    if successful_cnt_10 > 0:
        print('Covering Cost (byte=10): ', format(sum(covering_costs_10) / len(covering_costs_10), '.4f'))
        print('Success Rate (byte=10): ', format(successful_cnt_10 / (successful_cnt_10 + failed_cnt_10), '.4f'))
    if successful_cnt_20 > 0:
        print('Covering Cost (byte=20): ', format(sum(covering_costs_20) / len(covering_costs_20), '.4f'))
        print('Success Rate (byte=20): ', format(successful_cnt_20 / (successful_cnt_20 + failed_cnt_20), '.4f'))
    if successful_cnt_30 > 0:
        print('Covering Cost (byte=30): ', format(sum(covering_costs_30) / len(covering_costs_30), '.4f'))
        print('Success Rate (byte=30): ', format(successful_cnt_30 / (successful_cnt_30 + failed_cnt_30), '.4f'))


if __name__ == '__main__':
    byte_constraint = 20
    main()
