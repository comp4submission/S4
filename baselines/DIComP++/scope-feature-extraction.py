#!/usr/bin/python3

'''
scope-feature-extraction.py

Given a directory of ARM disassembly files from objdump,
(see Section 3 - Preprocessing in the paper), generates
a feature vector for each binary by making frequency
distributions for registers and calculating TF-IDF
scores for opcodes.

Outputs a csv file with features and ground truth labels
to be used by the SVM classifiers.
'''

import os
import re
import csv
import time
import pickle
import numpy as np

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer

# Disable scientific notation and round, for debugging
# We measure wall clock time here for logging only.
# For all runtime measurements in the paper, we use the time
# utility to sum the user and sys times.
start = time.time()
np.set_printoptions(precision=6, suppress=True)


objdumps_path = '/dev/shm/DIComP++'
prefix_for_output_files = 'features'

registers = ["rax", "rbx", "rcx", "rdx", "rdi", "rsi", "rsp", "rbp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15", "rip"]

files = os.listdir(objdumps_path)

opcode_strings = []  # holds sequence of opcodes for each binary to prep for TF-IDF
names = []  # binary names (contains ground truth labels to be used by classifier)
register_features = []  # holds frequency distributions for register use

outfile = open(f"{prefix_for_output_files}.csv", "w", newline='')
writer = csv.writer(outfile, delimiter=' ', quotechar='|',
                    quoting=csv.QUOTE_MINIMAL)

for fname in tqdm(files):

    names.append(fname)
    opcode_list = []

    # We profile register use separately for when each register is specified
    # as a destination operand vs. a source operand
    dst_reg_vec = np.zeros(len(registers))
    src_reg_vec = np.zeros(len(registers))

    with open(os.path.join(objdumps_path, fname), "r") as file:
        for line in file:
            line = line.replace("\n", "")

            # Objdump adds two spaces to lines with instructions
            if line[:2] != "  ":
                continue

            # Replace multiple whitespace with one space
            line = re.sub('[,;<>\[\]]', '', line)
            line = re.sub('\s+', ' ', line)
            split = line.split(' ')
            # empty string, instr address, opcode, args
            split = split[2:]

            # split: ['mov', 'x2', 'x1']
            # split: ['mov', 'w0', '#0x0', '//', '#0']
            # split: ['stp', 'x29', 'x30', 'sp', '#-16!']
            # Parse opcode
            opcode = split.pop(0)
            if not opcode.startswith('.'):
                opcode_list.append(opcode)

            # Parse args for registers
            for i in range(len(split)):
                try:
                    reg_idx = registers.index(split[i])
                    # First operand is destination
                    # All following operands are source 
                    if i == 0:
                        dst_reg_vec[reg_idx] += 1
                    else:
                        src_reg_vec[reg_idx] += 1
                except ValueError:
                    pass

    # Normalization (each distribution sums to 1)
    dst_reg_vec /= np.sum(dst_reg_vec)
    src_reg_vec /= np.sum(src_reg_vec)

    # Concatenate distributions into one register feature
    register_features.append(np.concatenate((dst_reg_vec, src_reg_vec)))
    opcode_strings.append(' '.join(opcode_list))

# TF-IDF scoring
vectorizer = TfidfVectorizer(stop_words=[])
X = vectorizer.fit_transform(opcode_strings).toarray()
assert (len(X) == len(names))

for i in range(len(names)):
    # The "dummy" is a remnant from older code whose unit of classification
    # was function, not binary. Leaving to maintain formatting consistency
    writer.writerow([names[i], "dummy"])
    writer.writerow(np.concatenate((register_features[i], X[i])))
outfile.close()

# Pickle the vocabulary and IDF vector
# For use in feature extraction of binaries with no ground truth for testing.
with open(f"{prefix_for_output_files}-vocab.pkl", "wb") as f:
    pickle.dump(vectorizer.vocabulary_, f)

with open(f"{prefix_for_output_files}-idf.pkl", "wb") as f:
    pickle.dump(vectorizer.idf_, f)

end = time.time()
with open(f"{prefix_for_output_files}.log", "a") as f:
    # Log outputs and elapsed time
    f.write(repr(vectorizer.vocabulary_))
    f.write("\n")
    f.write(repr(vectorizer.idf_))
    f.write("\n")
    f.write(f"Elapsed time in seconds: {round(end - start, 2)}\n")
    # This is less accurate than the user+sys time we measure
