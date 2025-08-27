import os
import shutil
import subprocess

from tqdm import tqdm

data_type = 'train'

ORIGINAL_DATA_PATH = f'/dev/shm/split_dataset/{data_type}'

NEW_DATA_PATH = f'/dev/shm/split_dataset/{data_type}_for_yara_opt'

EXCLUDE_FILE_SUFFIX = ['i64', 'idb', 'asm', 'id0', 'id1', 'id2', 'nam', 'til', '$$$', 'norm_code', 'fcg', 'icfg', 'json']


def main():
    count = 0
    for project_bin in tqdm(os.listdir(ORIGINAL_DATA_PATH)):
        for file in os.listdir(os.path.join(ORIGINAL_DATA_PATH, project_bin)):
            ext = file.split('.')[-1]
            if ext in EXCLUDE_FILE_SUFFIX:
                continue

            compiler, opt = file.split('_')

            os.makedirs(os.path.join(NEW_DATA_PATH, opt), exist_ok=True)
            shutil.copy(os.path.join(ORIGINAL_DATA_PATH, project_bin, file),
                        os.path.join(NEW_DATA_PATH, opt, f'{compiler}_{project_bin}'))
            subprocess.call(['/usr/bin/strip', '-g', '--remove-section=.comment',
                             os.path.join(NEW_DATA_PATH, opt, f'{compiler}_{project_bin}')],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            count += 1

    print('[INFO]', f'Copied {count} files and stripped them')


if __name__ == '__main__':
    main()
