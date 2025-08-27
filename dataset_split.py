import os
import shutil
import random
import argparse

from tqdm import tqdm


def get_args():
    arg_parser = argparse.ArgumentParser(
        description='This script is to split dataset into training, validation, and test sets.')
    arg_parser.add_argument('--dataset_path', type=str, default='/mnt/ssd1/anonymous/wo_debug/x86_64',
                            help='The original whole dataset')
    arg_parser.add_argument('--output_path', type=str, default='/dev/shm/split_dataset')
    arg_parser.add_argument('--seed', type=int, default=10)

    args = arg_parser.parse_args()
    return args


def main():
    args = get_args()

    if not os.path.exists(args.dataset_path):
        print('[ERROR]', 'Original Whole Dataset Not Found')
        return

    files = []
    for f in os.listdir(args.dataset_path):
        cnt_folder = os.path.join(args.dataset_path, f)
        if os.path.isdir(cnt_folder):
            files.append(cnt_folder)

    # train:val:test = 70:15:15
    val_num = int(len(files) * 0.15)
    test_num = int(len(files) * 0.15)
    train_num = len(files) - val_num - test_num

    assert (train_num + val_num + test_num) == len(files)

    output_train_path = os.path.join(args.output_path, 'train')
    #output_validation_path = os.path.join(args.output_path, 'validation')
    output_test_path = os.path.join(args.output_path, 'test')

    os.makedirs(output_train_path, exist_ok=True)
    #os.makedirs(output_validation_path, exist_ok=True)
    os.makedirs(output_test_path, exist_ok=True)

    random.seed(args.seed)
    random.shuffle(files)

    for i, f in enumerate(tqdm(files)):
        if i < train_num:
            shutil.copytree(f, os.path.join(output_train_path, os.path.basename(f)))
        elif i < (train_num + val_num):
            pass
            #shutil.copytree(f, os.path.join(output_validation_path, os.path.basename(f)))
        else:
            shutil.copytree(f, os.path.join(output_test_path, os.path.basename(f)))


if __name__ == '__main__':
    main()

