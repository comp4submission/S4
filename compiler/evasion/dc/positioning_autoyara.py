import os

import yara
import json


def get_correct_matched_binaries(rule, cnt_compiler):
    correct_matched_binaries = []

    fold_name = cnt_compiler.replace('_', '-', 1).replace('_', '.')

    for sample in os.listdir(os.path.join(TESTSET_PATH, fold_name)):
        if len(rule.match(os.path.join(TESTSET_PATH, fold_name, sample))) > 0:
            correct_matched_binaries.append(os.path.join(TESTSET_PATH, fold_name, sample))

    return correct_matched_binaries


def calculate_covering_cost(signature: tuple, idx_weight_map: dict[int, int]) -> float:
    # the idx list of byte of interest
    byte_of_interest = list(range(*signature))

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


def positioning(filepath, rules):
    global byte_constraint

    with open(filepath, 'rb') as f:
        original_bin_content: bytes = f.read()

    # get the byteweight of the original binary
    bin_name = os.path.basename(filepath)
    compiler = os.path.basename(os.path.dirname(filepath))
    opt, project, name = bin_name.split('_', maxsplit=2)
    byteweight_path = os.path.join('/mnt/ssd1/anonymous/wo_debug/x86_64', f'{project}_{name}', f'{compiler}_{opt}.byteweight.json')

    idx_weight_map = get_byteweight_from_json(byteweight_path, len(original_bin_content))

    original_scan_result: list[str] = scan_by_file_content(original_bin_content, rules)

    # initialize the minimum signature
    signatures = []
    worklist: list[tuple] = [(0, len(original_bin_content))]

    while len(worklist) > 0:
        cnt_interval = worklist.pop()

        # check if the scan result changes
        new_bin_content: bytes = patch(original_bin_content, cnt_interval)
        cnt_result = scan_by_file_content(new_bin_content, rules)

        if not is_equal(cnt_result, original_scan_result):

            if (cnt_interval[1] - cnt_interval[0]) <= byte_constraint:
                signatures.append(cnt_interval)
            else:
                # the cnt_interval is too long, split it
                worklist.extend(split(cnt_interval))

    # decide if the evasion is successful
    if len(signatures) == 0:
        print(f'Evasion Failed under Constraint={byte_constraint}')
        return None

    # initial minimium covering cost
    min_covering_cost = calculate_covering_cost(signatures[0], idx_weight_map)
    for sig in signatures:
        worklist = split(sig)
        while len(worklist) > 0:
            cnt_interval = worklist.pop()
            new_bin_content: bytes = patch(original_bin_content, cnt_interval)
            cnt_result = scan_by_file_content(new_bin_content, rules)

            if not is_equal(cnt_result, original_scan_result):

                cnt_covering_cost = calculate_covering_cost(cnt_interval, idx_weight_map)
                if cnt_covering_cost < min_covering_cost:
                    min_covering_cost = cnt_covering_cost

                worklist.extend(split(cnt_interval))

    return min_covering_cost


if __name__ == '__main__':

    TESTSET_PATH = '/dev/shm/split_dataset/test_for_yara_compiler'
    RULES_PATH = '/home/anonymous/AutoYara/compiler_rules'

    byte_constraint = 20

    min_covering_costs = []

    successful_cnt = 0
    failed_cnt = 0

    for rule_file in os.listdir(RULES_PATH):

        # get ground truth
        if rule_file.endswith('.yar'):
            compiler = rule_file[:-len('.yar')]
        else:
            compiler = rule_file

        compiled_rule = yara.compile(os.path.join(RULES_PATH, rule_file))

        # get the correctly matched binaries
        binaries = get_correct_matched_binaries(rule=compiled_rule, cnt_compiler=compiler)

        for binary_path in binaries:

            min_covering_cost = positioning(filepath=binary_path, rules=compiled_rule)

            if min_covering_cost is None:
                failed_cnt += 1
                continue

            # successful evasion
            successful_cnt += 1
            min_covering_costs.append(min_covering_cost)

            print('Minimum Covering Cost: ', format(sum(min_covering_costs) / len(min_covering_costs), '.4f'))
            print('Success Rate: ', format(successful_cnt / (successful_cnt + failed_cnt), '.4f'))

    print('Minimum Covering Cost: ', format(sum(min_covering_costs) / len(min_covering_costs), '.4f'))
    print('Success Rate: ', format(successful_cnt / (successful_cnt + failed_cnt), '.4f'))
