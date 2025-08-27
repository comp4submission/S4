#!/usr/bin/python3

"""
amd64-clf.py

Reads a csv file from the feature extraction step,
where every two rows consist of:
Row 1) <binary name> <function name>
Row 2) feature vector

First, parses binary name into label vector for:
1) Compiler family (gcc vs. Clang)
-- gcc = -1, Clang = +1
2) Optimization level (-O0, 1, 2, 3, s)
-- As 1, 2, 3, 4, 5 respectively

Then performs 10-fold cross validation for each label above.
"""

import sys
import csv
import itertools
import pickle
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC

if len(sys.argv) != 3:
    print(f"Usage: {sys.argv[0]} <feature csv> <vocabulary pickle>")
    exit(1)

# key = binary name
# value = set of function names of that binary
counter_dict = {}
counter_features = 0

bin_names = []  # for getting ground-truth labels
func_names = []  # not used for current model version

features = []
# For 64-bit ARM, classification tasks are compiler family
# and optimization level
label_cc_family = []
label_optim = []

with open(sys.argv[1], "r") as file:
    reader = csv.reader(file, delimiter=" ")
    line_num = 0

    # Hack to read two lines at a time
    for line1, line2 in itertools.zip_longest(*[reader] * 2):
        # Normalization can sometimes result in NaN for
        # some elements, which we skip
        if "nan" in line2:
            continue

        counter_features += 1
        bin_name, func_name = line1[0], line1[1]

        # For statistics - to be deprecated
        if bin_name not in counter_dict:
            counter_dict[bin_name] = {func_name}
        else:
            counter_dict[bin_name].add(func_name)

        # CSV is parsed as string - convert features to int
        bin_names.append(bin_name)
        func_names.append(func_name)

        feats = list(map(float, line2))
        features.append(feats)

        # Parse binary name to get compiler family labels
        if 'clang-4' in bin_name:
            label_cc_family.append(0)
        elif 'clang-5' in bin_name:
            label_cc_family.append(1)
        elif 'clang-6' in bin_name:
            label_cc_family.append(2)
        elif 'clang-7' in bin_name:
            label_cc_family.append(3)
        elif 'clang-8' in bin_name:
            label_cc_family.append(4)
        elif 'clang-9' in bin_name:
            label_cc_family.append(5)
        elif 'clang-10' in bin_name:
            label_cc_family.append(6)
        elif 'clang-11' in bin_name:
            label_cc_family.append(7)
        elif 'clang-12' in bin_name:
            label_cc_family.append(8)
        elif 'clang-13' in bin_name:
            label_cc_family.append(9)
        elif 'gcc-4' in bin_name:
            label_cc_family.append(10)
        elif 'gcc-5' in bin_name:
            label_cc_family.append(11)
        elif 'gcc-6' in bin_name:
            label_cc_family.append(12)
        elif 'gcc-7' in bin_name:
            label_cc_family.append(13)
        elif 'gcc-8' in bin_name:
            label_cc_family.append(14)
        elif 'gcc-9' in bin_name:
            label_cc_family.append(15)
        elif 'gcc-10' in bin_name:
            label_cc_family.append(16)
        elif 'gcc-11' in bin_name:
            label_cc_family.append(17)
        else:
            raise RuntimeError(f"Invalid binary name: {bin_name}")

        # Similarly, parse for optimization label
        if "-O0" in bin_name:
            label_optim.append(0)
        elif "-O1" in bin_name:
            label_optim.append(1)
        elif "-O2" in bin_name:
            label_optim.append(2)
        elif "-O3" in bin_name:
            label_optim.append(3)
        elif "-Os" in bin_name:
            label_optim.append(4)
        elif "-Ofast" in bin_name:
            label_optim.append(5)
        else:
            raise RuntimeError(f"Invalid binary name: {bin_name}")

# Create feature_names vector for ELI5.
# We need the vocabulary from the TF-IDF vectorizer
# to see which opcode is mapped to which dimension
vocab = pickle.load(open(sys.argv[2], "rb"))

opcode_feature_names = [''] * len(vocab)
for key, val in vocab.items():
    opcode_feature_names[val] = key

print(f"===================PARSED FEATURE CSV====================")
print(f"Number of binaries: {len(counter_dict)}")
print(f"Number of functions: {sum([len(v) for v in counter_dict.values()])}")
print(f"Number of features: {counter_features}")

# All lists are populated - convert to NumPy arrays
bin_names = np.array(bin_names)
func_names = np.array(func_names)
features = np.array(features)
label_cc_family = np.array(label_cc_family)
label_optim = np.array(label_optim)

# Remove duplicate feature vectors (denoise label space)
features, indices, counts = np.unique(features, axis=0, return_index=True, return_counts=True)

features = features[counts == 1]
bin_names = bin_names[indices[counts == 1]]
func_names = func_names[indices[counts == 1]]
label_cc_family = label_cc_family[indices[counts == 1]]
label_optim = label_optim[indices[counts == 1]]
print(f"Number of UNIQUE features: {len(features)}")

print(f"====================CV: COMPILER FAMILY====================")

print(features.shape, label_cc_family.shape, label_optim.shape)
X = features
y = label_cc_family
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# Pickle classifier trained on entire set 
compiler_clf = LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
compiler_clf.fit(X_train, y_train)

compiler_pred = compiler_clf.predict(X_test)
print(classification_report(y_test, compiler_pred, digits=4))

print(f"====================CV: OPTIMIZATION LEVEL====================")

y = label_optim
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=6)

# Pickle classifier trained on entire set 
opt_clf = LinearSVC(dual=False, penalty='l1', class_weight='balanced', max_iter=100000)
opt_clf.fit(X_train, y_train)
opt_pred = opt_clf.predict(X_test)
print(classification_report(y_test, opt_pred, target_names=['O0', 'O1', 'O2', 'O3', 'Os', 'Ofast'], digits=4))
