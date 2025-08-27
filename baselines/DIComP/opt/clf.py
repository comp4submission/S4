import joblib
import numpy as np

from sklearn.neural_network import MLPClassifier

from utils.evaluation import single_eval
from sklearn.metrics import classification_report
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


if __name__ == '__main__':
    TRAIN_DATA = joblib.load('/dev/shm/DIComP/opt/binaries_train.pkl')
    TEST_DATA = joblib.load('/dev/shm/DIComP/opt/binaries_test.pkl')
    main(TRAIN_DATA, TEST_DATA)
