import os
import subprocess

from tqdm import tqdm
from multiprocessing import Pool

TRAINSET_PATH = '/dev/shm/split_dataset/train_for_yara_compiler'

os.makedirs('compiler_rules', exist_ok=True)


def process(compiler):
    subprocess.call(['java', '-jar', 'AutoYara.jar',
                     '-b', f'{compiler}-negative-bytes', '-m', f'{compiler}-positive-bytes',
                     '-i', os.path.join(TRAINSET_PATH, compiler),
                     '-o', f'compiler_rules/{compiler.replace(".", "_").replace("-", "_")}'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def main():
    # generate yara rules in a mutiprocess way
    compilers = os.listdir(TRAINSET_PATH)

    with Pool(processes=32) as pool:
        for _ in tqdm(
                pool.imap_unordered(func=process, iterable=compilers),
                total=len(compilers)):
            pass


if __name__ == '__main__':
    main()
