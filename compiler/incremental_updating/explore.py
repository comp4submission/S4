import os
import copy
import joblib
import numpy as np

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


def vectorization(data: dict[str, dict[str, list[tuple]]], vectorizer=None, max_inst_seq_length=20000,
                  involved_compilers=None):
    corpus = []
    y = []

    for i, compiler in enumerate(involved_compilers):

        for inst_seqs in data[compiler].values():
            for inst_tuple in inst_seqs:
                inst_tuple = inst_tuple[:min(max_inst_seq_length, len(inst_tuple) - 1)]

                # opcode_list = [inst.split(',', maxsplit=1)[0] for inst in inst_tuple]
                # corpus.append(' '.join(opcode_list))
                corpus.append(' '.join(inst_tuple))
                y.append(i)

    if vectorizer is None:
        vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None)
        vectorizer.fit(corpus)

    X = vectorizer.transform(corpus).toarray()
    y = np.array(y)

    return X, y, vectorizer


def binary_vectorization(data: dict[str, dict[str, list[tuple]]], vectorizer=None, max_inst_seq_length=20000,
                         negative_compilers: list = None, positive_compilers: list = None):
    corpus = []
    y = []

    for compiler in negative_compilers + positive_compilers:

        for inst_seqs in data[compiler].values():
            for inst_tuple in inst_seqs:
                inst_tuple = inst_tuple[:min(max_inst_seq_length, len(inst_tuple) - 1)]

                # opcode_list = [inst.split(',', maxsplit=1)[0] for inst in inst_tuple]
                # corpus.append(' '.join(opcode_list))
                corpus.append(' '.join(inst_tuple))

                if compiler in negative_compilers:
                    y.append(0)
                else:
                    y.append(1)

    if vectorizer is None:
        vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None)
        vectorizer.fit(corpus)

    X = vectorizer.transform(corpus).toarray()
    y = np.array(y)

    return X, y, vectorizer


def main():
    sorted_compilers = ['gcc-4.9.4', 'clang-4.0', 'clang-5.0', 'gcc-5.5.0', 'gcc-7.3.0', 'clang-6.0',
                        'gcc-8.2.0', 'clang-7.0', 'gcc-6.5.0', 'clang-8.0', 'clang-9.0', 'clang-10.0',
                        'clang-11.0', 'gcc-10.3.0', 'clang-12.0', 'gcc-9.4.0', 'gcc-11.2.0', 'clang-13.0']

    training_pkl_path = '/dev/shm/compiler/normalized_instruction/binaries_train.pkl'
    testing_pkl_path  = '/dev/shm/compiler/normalized_instruction/binaries_test.pkl'

    train_data: dict = joblib.load(training_pkl_path)
    test_data: dict = joblib.load(testing_pkl_path)
    max_inst_seq_length = 20000

    for compiler in sorted_compilers:
        assert compiler in train_data.keys() and compiler in test_data.keys()

    # now start incremental experiments

    result = {}

    incremental_trained_classifiers = {}
    incremental_trained_vectorizers = {}

    for i in range(2, len(sorted_compilers) + 1):

        full_training_part: list[str] = sorted_compilers[:i]
        incremental_training_part: list[str] = sorted_compilers[i:]

        print('Full training classifiers:', full_training_part)
        # the first i classifiers are trained in the full training mode
        # and the remaining len(sorted_compilers)-i classifiers are trained in the incremental training mode

        training_data_for_full_training = {c: train_data[c] for c in full_training_part}
        X_train, y_train, vectorizer = vectorization(training_data_for_full_training,
                                                     max_inst_seq_length=max_inst_seq_length,
                                                     involved_compilers=full_training_part)

        full_training_classifier = OneVsRestClassifier(XGBClassifier(nthread=int(os.cpu_count() * 0.9)))
        full_training_classifier.fit(X_train, y_train)
        full_training_vectorizer = copy.deepcopy(vectorizer)

        # train the remaining classifiers in the incremental mode
        for j in range(i, len(sorted_compilers)):
            cnt_classifier = sorted_compilers[j]
            print(f'Classifier for {cnt_classifier} is trained in the incremental training mode')
            if cnt_classifier not in incremental_trained_classifiers.keys():
                # construct the training data
                X_train, y_train, vectorizer = binary_vectorization(train_data, max_inst_seq_length=max_inst_seq_length,
                                                                    negative_compilers=sorted_compilers[:j],
                                                                    positive_compilers=[cnt_classifier])
                model = XGBClassifier(nthread=int(os.cpu_count() * 0.9))
                model.fit(X_train, y_train)

                incremental_trained_classifiers[cnt_classifier] = copy.deepcopy(model)
                incremental_trained_vectorizers[cnt_classifier] = copy.deepcopy(vectorizer)

        # evaluation

        # key: compiler name, value: the prediction of all test samples [0,0,0,1,0,1,0,0,...]
        all_preds = {}

        # first, get the result from the full training classifier
        X_test, y_test, _ = vectorization(test_data, vectorizer=full_training_vectorizer,
                                          max_inst_seq_length=max_inst_seq_length,
                                          involved_compilers=sorted_compilers)

        preds_by_full_training_classifier = full_training_classifier.predict(X_test)

        # format the prediction of full training classifier
        # transform the result [3, 4] to the format of [[0,0,0,1,0],[0,0,0,0,1]]
        for c in full_training_part:
            all_preds[c] = [0] * len(X_test)

        for j, class_id in enumerate(preds_by_full_training_classifier):
            all_preds[full_training_part[class_id]][j] = 1

        # then, get the result from the incremental training classifiers
        for c in incremental_training_part:
            # the test data for the j-th incremental trained classifier
            X_test, y_test, _ = vectorization(test_data, vectorizer=incremental_trained_vectorizers[c],
                                              max_inst_seq_length=max_inst_seq_length,
                                              involved_compilers=sorted_compilers)

            all_preds[c] = incremental_trained_classifiers[c].predict(X_test)

        # get the final decision
        final_preds = []
        for j in range(len(X_test)):

            # the prediction for the j-th test sample
            for c in reversed(sorted_compilers):
                if all_preds[c][j] == 1:
                    final_preds.append(sorted_compilers.index(c))
                    break

        p = precision_score(y_test, final_preds, average='macro')
        r = recall_score(y_test, final_preds, average='macro')
        f1 = f1_score(y_test, final_preds, average='macro')
        acc = accuracy_score(y_test, final_preds)

        print('i =', i, p, r, f1, acc)

        result[i] = (p, r, f1, acc)

    joblib.dump(result, 'result.pkl')


if __name__ == '__main__':
    main()
