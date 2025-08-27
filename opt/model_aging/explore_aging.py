import os
import pickle
import joblib
import numpy as np

from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer


def vectorization(data: dict[str, list[tuple]], vectorizer=None, max_inst_seq_length=20000):
    corpus = []
    y = []

    for i, opt in enumerate(sorted(data.keys())):

        for inst_tuple in data[opt]:
            inst_tuple: tuple[str, ...]

            inst_tuple = inst_tuple[:min(max_inst_seq_length, len(inst_tuple) - 1)]

            # opcode_list = [inst.split(',', maxsplit=1)[0] for inst in inst_tuple]
            corpus.append(' '.join(inst_tuple))
            y.append(i)

    if vectorizer is None:
        # the current data is training set
        vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None)
        vectorizer.fit(corpus)

    X = vectorizer.transform(corpus).toarray()
    y = np.array(y)

    return X, y, vectorizer


def eval_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    # make predictions
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), f1_score(y_test, preds, average='macro')


def main(ovr, compiler_family, compilers, max_inst_seq_length):
    # training_pkl_path = '/mnt/ssd1/anonymous/binary_level_compiler_dataset/normalized_instruction/binaries_train.pkl'
    # testing_pkl_path = '/mnt/ssd1/anonymous/binary_level_compiler_dataset/normalized_instruction/binaries_test.pkl'

    training_pkl_path = '/dev/shm/compiler/normalized_instruction/binaries_train.pkl'
    testing_pkl_path  = '/dev/shm/compiler/normalized_instruction/binaries_test.pkl'

    print('OvR Mode:', ovr)

    # original dataset
    training_compiler_opt_dict: dict = joblib.load(training_pkl_path)
    testing_compiler_opt_dict: dict = joblib.load(testing_pkl_path)
    print('Whole Dataset Loaded')

    result_acc = {}
    result_f1 = {}

    for num_of_cross_version in range(8):
        for num_of_cumulated in range(1, len(compilers) - num_of_cross_version + 1):
            training_compilers = compilers[:num_of_cumulated]
            testing_compiler = compilers[num_of_cumulated + num_of_cross_version - 1]

            print(num_of_cross_version, num_of_cumulated)
            print(training_compilers, testing_compiler)

            # transform the dataset format
            train_data = {}
            test_data = {}
            for compiler, opt_dict in training_compiler_opt_dict.items():

                if compiler not in training_compilers:
                    continue

                for opt, inst_seq_list in opt_dict.items():
                    if opt not in train_data.keys():
                        train_data[opt] = inst_seq_list
                    else:
                        train_data[opt].extend(inst_seq_list)

            for compiler, opt_dict in testing_compiler_opt_dict.items():

                if compiler != testing_compiler:
                    continue

                for opt, inst_seq_list in opt_dict.items():
                    if opt not in test_data.keys():
                        test_data[opt] = inst_seq_list
                    else:
                        test_data[opt].extend(inst_seq_list)

            print('Dataset Prepared')

            X_train, y_train, vectorizer = vectorization(train_data, max_inst_seq_length=max_inst_seq_length)
            X_test, y_test, _ = vectorization(test_data, vectorizer=vectorizer, max_inst_seq_length=max_inst_seq_length)

            if ovr:
                model = OneVsRestClassifier(XGBClassifier(nthread=os.cpu_count() // 2))
            else:
                model = XGBClassifier(nthread=os.cpu_count() // 2)

            acc, f1 = eval_model(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

            print('Model Trained')

            if num_of_cumulated not in result_acc.keys():
                result_acc[num_of_cumulated] = {}
            result_acc[num_of_cumulated][num_of_cross_version] = acc

            if num_of_cumulated not in result_f1.keys():
                result_f1[num_of_cumulated] = {}
            result_f1[num_of_cumulated][num_of_cross_version] = f1

    pickle.dump(result_acc, open(f'result_acc_{compiler_family}_ovr_{ovr}.pkl', 'wb'))
    pickle.dump(result_f1, open(f'result_f1_{compiler_family}_ovr_{ovr}.pkl', 'wb'))


if __name__ == '__main__':

    gcc_compilers = ['gcc-4.9.4', 'gcc-5.5.0', 'gcc-6.5.0', 'gcc-7.3.0',
                     'gcc-8.2.0', 'gcc-9.4.0', 'gcc-10.3.0', 'gcc-11.2.0']
    clang_compilers = ['clang-4.0', 'clang-5.0', 'clang-6.0', 'clang-7.0', 'clang-8.0',
                       'clang-9.0', 'clang-10.0', 'clang-11.0', 'clang-12.0', 'clang-13.0']

    max_inst_seq_length = 20000
    main(ovr=True, compiler_family='gcc', compilers=gcc_compilers, max_inst_seq_length=max_inst_seq_length)
    main(ovr=False, compiler_family='gcc', compilers=gcc_compilers, max_inst_seq_length=max_inst_seq_length)
    main(ovr=True, compiler_family='clang', compilers=clang_compilers, max_inst_seq_length=max_inst_seq_length)
    main(ovr=False, compiler_family='clang', compilers=clang_compilers, max_inst_seq_length=max_inst_seq_length)
