import os
import yara
import json
import pygad
import random
import numpy as np

from tqdm import tqdm


def get_correct_matched_binaries(rule, cnt_compiler):
    correct_matched_binaries = []

    fold_name = cnt_compiler.replace('_', '-', 1).replace('_', '.')

    for sample in os.listdir(os.path.join(TESTSET_PATH, fold_name)):
        if len(rule.match(os.path.join(TESTSET_PATH, fold_name, sample))) > 0:
            correct_matched_binaries.append(os.path.join(TESTSET_PATH, fold_name, sample))

    return correct_matched_binaries


def calculate_covering_cost_by_boi(byte_of_interest, idx_weight_map: dict[int, int]) -> float:

    mean = sum(byte_of_interest) / len(byte_of_interest)

    covering_cost = 0
    for index in byte_of_interest:
        covering_cost += idx_weight_map[index] * ((index - mean)**2)

    return covering_cost



def scan_by_file_content(bin_content, rules) -> list[str]:
    matched_rules: list[str] = rules.match(data=bin_content)
    return matched_rules


def split(original_interval: tuple) -> list[tuple]:
    """
    Split the interval in two parts.
    """
    start, end = original_interval
    middle = (start + end) // 2

    if start == middle or end == middle:
        return []

    return [(start, middle), (middle, end)]


def patch(original_binary: bytes, interval: tuple, patch_byte=b'\x00') -> bytes:
    """
    Patch the binary with the new content.
    """
    start, end = interval
    new_content = original_binary[:start] + patch_byte * (end - start) + original_binary[end:]
    assert len(new_content) == len(original_binary)
    return new_content


def is_equal(l1, l2):
    return set(l1) == set(l2)


def is_evade(perturbation):
    global original_scan_result, original_bin_content, rules
    # perturbation: [0, 0, 0, 1, 0, 1, ...]
    new_bin_content = [original_bin_content[i] if perturbation[i] == 0 else 0 for i in range(len(perturbation))]
    new_bin_content = bytes(new_bin_content)
    new_result = scan_by_file_content(new_bin_content, rules)

    # scan result changed
    # True: evade
    # False: not evade
    return not is_equal(new_result, original_scan_result)


def get_byteweight_from_json(json_path: str, bin_size: int):

    # load the byteweight json and convert the key type to int
    with open(json_path) as f:
        idx_weight_map = json.load(f)
    idx_weight_map: dict[int, int] = {int(k): v for k, v in idx_weight_map.items()}

    assert max(idx_weight_map.keys()) <= bin_size

    # fill the missing indexes with weight 1
    for i in range(bin_size):
        if i not in idx_weight_map:
            idx_weight_map[i] = 1

    return idx_weight_map


def callback_generation(ga_instance):
    print(f"Changed bytes: {sum(ga_instance.best_solution()[0])}")
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Fitness    = {ga_instance.best_solution()[1]}")


def fitness_func(ga_instance, solution, solution_idx):
    fitness = is_evade(solution) + (1.0 / (sum(solution) + 2))
    return fitness


def positioning(filepath):

    global original_scan_result, original_bin_content, rules

    with open(filepath, 'rb') as f:
        original_bin_content = f.read()

    num_generations = 10000  # Number of generations.
    num_parents_mating = 20  # Number of solutions to be selected as parents in the mating pool.
    sol_per_pop = 50  # Number of solutions in the population.
    num_genes = len(original_bin_content)

    # get the byteweight of the original binary
    bin_name = os.path.basename(filepath)
    compiler = os.path.basename(os.path.dirname(filepath))
    opt, project, name = bin_name.split('_', maxsplit=2)
    byteweight_path = os.path.join('/mnt/ssd1/anonymous/wo_debug/x86_64', f'{project}_{name}', f'{compiler}_{opt}.byteweight.json')

    idx_weight_map = get_byteweight_from_json(byteweight_path, len(original_bin_content))

    original_scan_result = scan_by_file_content(original_bin_content, rules)

    initial_population = np.zeros(shape=(sol_per_pop, num_genes), dtype=int)
    # Creating an instance of the GA class inside the ga module. Some parameters are initialized within the constructor.
    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=fitness_func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           initial_population=initial_population,
                           mutation_percent_genes=0.1,
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
        modification = [0 for _ in range(len(original_bin_content))]

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
                            stop_criteria='saturate_20',
                            save_solutions=True)
    ga_instance2.run()

    best_partial_solution = ga_instance2.best_solution()[0]

    print(f"Changed bytes: {sum(best_partial_solution)}")

    adv_modification = [0 for _ in range(len(original_bin_content))]

    for i in range(len(best_partial_solution)):
        if best_partial_solution[i] == 1:
            adv_modification[adversarial_solution_idx[i]] = 1

    adv_modification = np.array(adv_modification)
    boi = np.where(adv_modification == 1)[0]
    covering_cost = calculate_covering_cost_by_boi(boi, idx_weight_map)

    return sum(best_partial_solution), covering_cost


if __name__ == '__main__':

    TESTSET_PATH = '/dev/shm/split_dataset/test_for_yara_compiler'
    RULES_PATH = '/home/anonymous/AutoYara/compiler_rules'

    original_scan_result = None
    original_bin_content = None
    rules = None

    covering_costs_10 = []
    covering_costs_20 = []
    covering_costs_30 = []

    successful_cnt_10 = 0
    failed_cnt_10 = 0

    successful_cnt_20 = 0
    failed_cnt_20 = 0

    successful_cnt_30 = 0
    failed_cnt_30 = 0

    for rule_file in os.listdir(RULES_PATH):

        # get ground truth
        if rule_file.endswith('.yar'):
            compiler = rule_file[:-len('.yar')]
        else:
            compiler = rule_file

        compiled_rule = yara.compile(os.path.join(RULES_PATH, rule_file))

        # get the correctly matched binaries
        binaries = get_correct_matched_binaries(rule=compiled_rule, cnt_compiler=compiler)

        # sampling for testing
        binaries = random.sample(binaries, int(0.1 * len(binaries)))

        for binary_path in tqdm(binaries):
            rules = compiled_rule
            num_changed_bytes, covering_cost = positioning(filepath=binary_path)

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

            if successful_cnt_10 > 0:
                print('Covering Cost (byte=10): ', format(sum(covering_costs_10) / len(covering_costs_10), '.4f'))
                print('Success Rate (byte=10): ', format(successful_cnt_10 / (successful_cnt_10 + failed_cnt_10), '.4f'))
            if successful_cnt_20 > 0:
                print('Covering Cost (byte=20): ', format(sum(covering_costs_20) / len(covering_costs_20), '.4f'))
                print('Success Rate (byte=20): ', format(successful_cnt_20 / (successful_cnt_20 + failed_cnt_20), '.4f'))
            if successful_cnt_30 > 0:
                print('Covering Cost (byte=30): ', format(sum(covering_costs_30) / len(covering_costs_30), '.4f'))
                print('Success Rate (byte=30): ', format(successful_cnt_30 / (successful_cnt_30 + failed_cnt_30), '.4f'))

    if successful_cnt_10 > 0:
        print('Covering Cost (byte=10): ', format(sum(covering_costs_10) / len(covering_costs_10), '.4f'))
        print('Success Rate (byte=10): ', format(successful_cnt_10 / (successful_cnt_10 + failed_cnt_10), '.4f'))
    if successful_cnt_20 > 0:
        print('Covering Cost (byte=20): ', format(sum(covering_costs_20) / len(covering_costs_20), '.4f'))
        print('Success Rate (byte=20): ', format(successful_cnt_20 / (successful_cnt_20 + failed_cnt_20), '.4f'))
    if successful_cnt_30 > 0:
        print('Covering Cost (byte=30): ', format(sum(covering_costs_30) / len(covering_costs_30), '.4f'))
        print('Success Rate (byte=30): ', format(successful_cnt_30 / (successful_cnt_30 + failed_cnt_30), '.4f'))
