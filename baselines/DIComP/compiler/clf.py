import joblib
import numpy as np

from sklearn.neural_network import MLPClassifier

from utils.evaluation import single_eval
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer


def vectorization(data: dict[str, dict[str, list[tuple]]], vectorizer=None, max_inst_seq_length=20000):
    corpus = []
    y = []

    for i, compiler in enumerate(sorted(data.keys())):

        for inst_seqs in data[compiler].values():
            for inst_tuple in inst_seqs:

                inst_tuple = inst_tuple[:min(max_inst_seq_length, len(inst_tuple) - 1)]

                corpus.append(' '.join(inst_tuple))
                y.append(i)

    if vectorizer is None:
        vectorizer = CountVectorizer(ngram_range=(1, 1), tokenizer=lambda x: x.split(' '), token_pattern=None)
        vectorizer.fit(corpus)
        print(f'Vocab Size: {len(vectorizer.get_feature_names_out())}')

    X = vectorizer.transform(corpus).toarray()
    y = np.array(y)

    return X, y, vectorizer


def main(train_data, test_data):

    target_names = sorted(train_data.keys())

    max_inst_seq_length = 32000

    X_train, y_train, vectorizer = vectorization(train_data, max_inst_seq_length=max_inst_seq_length)
    X_test, y_test, _ = vectorization(test_data, vectorizer=vectorizer, max_inst_seq_length=max_inst_seq_length)

    model = MLPClassifier()
    acc, preds = single_eval(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

    print(classification_report(y_test, preds, target_names=target_names, digits=4))

    joblib.dump(model, f'model_{max_inst_seq_length}.dat')
    joblib.dump(vectorizer.vocabulary_, f'vectorizer_vocab_{max_inst_seq_length}.dat')



if __name__ == '__main__':
    TRAIN_DATA = joblib.load('/dev/shm/DIComP/compiler/binaries_train.pkl')
    TEST_DATA = joblib.load('/dev/shm/DIComP/compiler/binaries_test.pkl')
    main(TRAIN_DATA, TEST_DATA)
