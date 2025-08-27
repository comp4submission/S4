import os
import joblib
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score


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
        print(f'Vocab Size: {len(vectorizer.get_feature_names_out())}')

    X = vectorizer.transform(corpus).toarray()
    y = np.array(y)

    return X, y, vectorizer


def main():
    test_data = joblib.load('binaries_test.pkl')

    model_filepath = 'model_42000.dat'

    target_names = sorted(test_data.keys())
    model = joblib.load(model_filepath)

    max_inst_seq_length = int(os.path.basename(model_filepath).split('_')[1].split('.')[0])

    vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None,
                                 vocabulary=joblib.load('vectorizer_vocab.dat'))

    X_test, y_test, _ = vectorization(test_data, vectorizer=vectorizer, max_inst_seq_length=max_inst_seq_length)

    preds = model.predict(X_test)

    print('Accuracy: ', accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=target_names, digits=4))


if __name__ == '__main__':
    main()
