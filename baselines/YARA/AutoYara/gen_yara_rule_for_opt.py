import os
import subprocess

from tqdm import tqdm
from multiprocessing import Pool

TRAINSET_PATH = '/dev/shm/split_dataset/train_for_yara_opt'

os.makedirs('opt_rules', exist_ok=True)


def process(opt):
    subprocess.call(['java', '-jar', 'AutoYara.jar',
                     '-b', f'{opt}-negative-bytes', '-m', f'{opt}-positive-bytes',
                     '-i', os.path.join(TRAINSET_PATH, opt),
                     '-o', f'opt_rules/{opt}'],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def main():
    # generate yara rules in a mutiprocess way
    opts = os.listdir(TRAINSET_PATH)

    with Pool(processes=32) as pool:
        for _ in tqdm(
                pool.imap_unordered(func=process, iterable=opts),
                total=len(opts)):
            pass


if __name__ == '__main__':
    main()
