import os
import shutil
import subprocess

TRAINSET_PATH = '/dev/shm/split_dataset/train_for_yara_opt'

os.makedirs('opt_rules', exist_ok=True)


def main():
    for opt in os.listdir(TRAINSET_PATH):
        # delete all dbs
        for db in os.listdir('dbs'):
            os.remove(os.path.join('dbs', db))
        # prepare the good dbs
        for good_db in os.listdir('all_dbs'):
            if 'opt' in good_db and opt not in good_db:
                shutil.copy(os.path.join('all_dbs', good_db), os.path.join('dbs', good_db))
        # generate yara rules
        subprocess.call(['python', 'yarGen.py', '-m', os.path.join(TRAINSET_PATH, opt),
                         '-o', os.path.join('opt_rules', f'{opt}.yar'),
                         '--opcodes', '--nofilesize', '--nomagic', '--noextras'])


if __name__ == '__main__':
    main()
