import os
import shutil
import subprocess

from tqdm import tqdm
from multiprocessing import Pool

TRAINSET_PATH = '/dev/shm/split_dataset/train_for_yara_opt'


def layout_positive_and_negative_samples(positive):
    if positive not in os.listdir(TRAINSET_PATH):
        print('Error: positive samples not found')
        exit(1)

    positive_path = os.path.join(os.path.dirname(TRAINSET_PATH), positive)
    shutil.move(os.path.join(TRAINSET_PATH, positive), positive_path)
    negative_path = TRAINSET_PATH
    return positive_path, negative_path


def process(argument):
    i, o, n = argument
    subprocess.call(['java', '-cp', 'AutoYara.jar', 'edu.lps.acs.ml.autoyara.Bytes2Bloom',
                     '-i', i,
                     '-o', o,
                     '-k', str(100000),
                     '-n', n],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return 0


def main():
    for opt in os.listdir(TRAINSET_PATH):
        # mkdir for the bloom filter
        os.makedirs(f'{opt}-positive-bytes', exist_ok=True)
        os.makedirs(f'{opt}-negative-bytes', exist_ok=True)

        # layout the positive and negative samples
        positive_path, negative_path = layout_positive_and_negative_samples(opt)

        # construct the bloom filter in a multiprocess way
        arguments_list = []
        # construct the arguments
        for n in [8, 16, 32, 64, 128, 256, 512, 1024]:
            arguments_list.append((positive_path, f'{opt}-positive-bytes', str(n)))
            arguments_list.append((negative_path, f'{opt}-negative-bytes', str(n)))

        with Pool(processes=16) as pool:
            for _ in tqdm(
                    pool.imap_unordered(func=process, iterable=arguments_list),
                    total=len(arguments_list)):
                pass

        # reset the layout
        shutil.move(positive_path, os.path.join(TRAINSET_PATH, opt))


if __name__ == '__main__':
    main()
