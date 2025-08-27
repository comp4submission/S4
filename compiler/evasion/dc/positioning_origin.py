import os
import math
import json
import random
import tempfile
import subprocess

from tqdm import tqdm
from multiprocessing import Pool
from elftools.elf.elffile import ELFFile

ORIGIN_PYTHON_PATH = '/home/anonymous/anaconda3/envs/origin/bin/python'


def generator(full_list, sublist_length: int):
    group_count = math.ceil(len(full_list) / sublist_length)

    for idx in range(group_count):
        start = sublist_length * idx
        end = min(start + sublist_length, len(full_list))

        yield full_list[start:end]


def is_elf_file(file_path: str):
    with open(file_path, 'rb') as f:
        return f.read(4) == b'\x7fELF'


def get_text_section_range(filepath) -> tuple[int, int]:
    with open(filepath, 'rb') as f:
        elffile = ELFFile(f)
        for section in elffile.iter_sections():
            if section.name.lower() == '.text':
                offset = section.header['sh_offset']
                size = section.header['sh_size']
                return offset, offset + size


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


def scan(bin_path):
    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            raw_result = subprocess.check_output([ORIGIN_PYTHON_PATH, 'Origin.py',
                                                  '--binpath', bin_path,
                                                  '--workingdir', tmp_dir,
                                                  '--modeldir', '/home/anonymous/origin/toolchain-origin/build/bin/data',
                                                  '--installdir', '/home/anonymous/origin/toolchain-origin/build'],
                                                 stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            return ''

        raw_result = raw_result.decode('utf-8')

        default_result = ''
        for line in raw_result.split('\n'):
            if line.startswith('Address range'):
                return line
        return default_result


def calculate_covering_cost(signature: tuple, idx_weight_map: dict[int, int]) -> float:
    # the idx list of byte of interest
    byte_of_interest = list(range(*signature))

    mean = sum(byte_of_interest) / len(byte_of_interest)

    covering_cost = 0
    for index in byte_of_interest:
        covering_cost += idx_weight_map[index] * ((index - mean)**2)

    return covering_cost


def scan_by_file_content(bin_content):
    with tempfile.NamedTemporaryFile() as tmp_file:
        tmp_file.write(bin_content)
        tmp_file.flush()

        result = scan(tmp_file.name)
        return result


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


def positioning(bin_path):

    global byte_constraint

    with open(bin_path, 'rb') as f:
        original_bin_content: bytes = f.read()

    original_scan_result = scan(bin_path)

    if original_scan_result == '':
        return None

    byteweight_path = f'{bin_path}.byteweight.json'
    idx_weight_map = get_byteweight_from_json(byteweight_path, len(original_bin_content))

    text_interval = get_text_section_range(bin_path)

    signatures = []
    worklist: list[tuple] = [text_interval]

    while len(worklist) > 0:
        cnt_interval = worklist.pop()

        # check if the scan result changes
        new_bin_content: bytes = patch(original_bin_content, cnt_interval)
        cnt_result = scan_by_file_content(new_bin_content)

        if cnt_result != original_scan_result:
            # the cnt_interval contains signature
            print('Signature in the interval: ', cnt_interval)

            if (cnt_interval[1] - cnt_interval[0]) <= byte_constraint:
                signatures.append(cnt_interval)
            else:
                # the cnt_interval is too long, split it
                worklist.extend(split(cnt_interval))

    # decide if the evasion is successful
    if len(signatures) == 0:
        print(f'Evasion Failed under Constraint={byte_constraint}')
        return None

    # refine each signature, and find the shortest refined signature

    min_covering_cost = calculate_covering_cost(signatures[0], idx_weight_map)


    for sig in signatures:
        worklist = split(sig)
        while len(worklist) > 0:
            cnt_interval = worklist.pop()
            new_bin_content: bytes = patch(original_bin_content, cnt_interval)
            cnt_result = scan_by_file_content(new_bin_content)

            if cnt_result != original_scan_result:
                cnt_covering_cost = calculate_covering_cost(cnt_interval, idx_weight_map)

                if cnt_covering_cost < min_covering_cost:
                    min_covering_cost = cnt_covering_cost

                worklist.extend(split(cnt_interval))

    return min_covering_cost


def batch_process(bin_paths):
    min_covering_costs = []
    successful_cnt = 0
    failed_cnt = 0

    for bin_path in bin_paths:

        min_covering_cost = positioning(bin_path)

        if min_covering_cost is None:
            failed_cnt += 1
            continue

        #  successful evasion
        successful_cnt += 1
        min_covering_costs.append(min_covering_cost)

    return min_covering_costs, successful_cnt, failed_cnt


def main():
    TESTSET_PATH = '/dev/shm/split_dataset/test'

    bin_paths = []
    for bin_dir in os.listdir(TESTSET_PATH):
        bin_dir_path = os.path.join(TESTSET_PATH, bin_dir)
        if not os.path.isdir(bin_dir_path):
            continue

        for bin_file in os.listdir(bin_dir_path):
            bin_file_path = os.path.join(bin_dir_path, bin_file)
            if is_elf_file(bin_file_path):
                bin_paths.append(bin_file_path)

    print('Total number of binaries: ', len(bin_paths))

    workload = 2
    group_count = math.ceil(len(bin_paths) / workload)

    min_covering_costs = []
    successful_cnt = 0
    failed_cnt = 0

    with Pool(processes=os.cpu_count() // 3) as pool:
        for sub_min_covering_costs, sub_successful_cnt, sub_failed_cnt in tqdm(
                pool.imap_unordered(func=batch_process, iterable=generator(bin_paths, workload)),
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
