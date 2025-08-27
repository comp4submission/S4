import os
import time
import joblib
import numpy as np

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score


def vectorization(data: dict[str, dict[str, list[tuple]]], vectorizer=None, max_inst_seq_length=20000):
    corpus = []
    y = []

    for i, compiler in enumerate(sorted(data.keys())):

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
        print(f'Vocab Size: {len(vectorizer.get_feature_names_out())}')

    X = vectorizer.transform(corpus).toarray()
    y = np.array(y)

    return X, y, vectorizer


def single_eval(model, X_train, X_test, y_train, y_test):
    print('Start Single Evaluation')
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    # make predictions
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), preds


def main(train_data, test_data):

    compiler_to_id = dict(zip(sorted(train_data.keys()), range(len(train_data.keys()))))
    joblib.dump(compiler_to_id, 'compiler_to_id.dat')

    target_names = sorted(train_data.keys())

    max_inst_seq_length = 32000

    X_train, y_train, vectorizer = vectorization(train_data, max_inst_seq_length=max_inst_seq_length)
    X_test, y_test, _ = vectorization(test_data, vectorizer=vectorizer, max_inst_seq_length=max_inst_seq_length)

    models = [SVC(kernel='rbf'), DecisionTreeClassifier(), RandomForestClassifier(), MLPClassifier(),
              XGBClassifier(nthread=os.cpu_count() // 3)]

    for model in models:
        print(model)
        acc, preds = single_eval(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
        print(classification_report(y_test, preds, target_names=target_names, digits=4))


if __name__ == '__main__':
    TRAIN_DATA = joblib.load('binaries_train.pkl')
    TEST_DATA = joblib.load('binaries_test.pkl')
    main(TRAIN_DATA, TEST_DATA)
