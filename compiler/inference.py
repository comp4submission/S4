import os
import joblib
import numpy as np

from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier
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


def main():
    test_data = joblib.load('binaries_test.pkl')

    model_file = 'model_32000.dat'

    max_inst_seq_length = int(os.path.basename(model_file).split('.')[0].split('_')[-1])

    print(max_inst_seq_length)

    model = joblib.load(model_file)
    compiler_to_id = joblib.load('compiler_to_id.dat')

    target_names = sorted(test_data.keys())

    vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None,
                                 vocabulary=joblib.load('vectorizer_vocab.dat'))

    X_test, y_test, _ = vectorization(test_data, vectorizer=vectorizer, max_inst_seq_length=max_inst_seq_length)

    preds = model.predict(X_test)

    print(classification_report(y_test, preds, target_names=target_names, digits=4))


if __name__ == '__main__':
    main()
