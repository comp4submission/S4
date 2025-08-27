import os
import shutil

from tqdm import tqdm

# if is_move is True, the original files will be moved to the new location, else they will be copied
is_move = True

ORIGINAL_BINKIT_PATH = "/dev/shm/BinKit_normal"
REARRANGED_BINKIT_PATH = "/dev/shm/BinKit_rearranged"


def main():
    for proj_wo_version in tqdm(os.listdir(ORIGINAL_BINKIT_PATH)):
        for binary in os.listdir(os.path.join(ORIGINAL_BINKIT_PATH, proj_wo_version)):
            project, compiler, arch, bit, opt, bin_name = binary.split('_', maxsplit=5)
            os.makedirs(os.path.join(REARRANGED_BINKIT_PATH, f'{arch}_{bit}'), exist_ok=True)
            os.makedirs(os.path.join(REARRANGED_BINKIT_PATH, f'{arch}_{bit}', f'{project}_{bin_name}'), exist_ok=True)
            new_bin_name = f'{compiler}_{opt}'
            if is_move is True:
                shutil.move(
                    os.path.join(ORIGINAL_BINKIT_PATH, proj_wo_version, binary),
                    os.path.join(REARRANGED_BINKIT_PATH, f'{arch}_{bit}', f'{project}_{bin_name}', new_bin_name)
                )
            else:
                shutil.copy(
                    os.path.join(ORIGINAL_BINKIT_PATH, proj_wo_version, binary),
                    os.path.join(REARRANGED_BINKIT_PATH, f'{arch}_{bit}', f'{project}_{bin_name}', new_bin_name)
                )


if __name__ == '__main__':
    main()
