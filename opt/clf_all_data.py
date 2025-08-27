import os
import joblib
import numpy as np

from tqdm import tqdm

from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

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
        vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None)
        vectorizer.fit(corpus)
        print(f'Vocab Size: {len(vectorizer.get_feature_names_out())}')

    X = vectorizer.transform(corpus).toarray()
    y = np.array(y)

    return X, y, vectorizer


def main(train_data, test_data):

    opt_to_id = dict(zip(sorted(train_data.keys()), range(len(train_data.keys()))))
    joblib.dump(opt_to_id, 'opt_to_id.dat')

    target_names = sorted(train_data.keys())

    best_acc = 0
    best_y_test = None
    best_y_pred = None

    last_model_path = None
    last_vocab_path = None

    for max_inst_seq_length in tqdm(range(10000, 50001, 2000)):
        X_train, y_train, vectorizer = vectorization(train_data, max_inst_seq_length=max_inst_seq_length)
        X_test, y_test, _ = vectorization(test_data, vectorizer=vectorizer, max_inst_seq_length=max_inst_seq_length)

        model = OneVsRestClassifier(XGBClassifier(nthread=os.cpu_count() // 3))
        acc, preds = single_eval(model=model, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        if acc > best_acc:
            print(f'Best Accuracy Updated: {acc} @ max_inst_seq_length={max_inst_seq_length}')
            best_acc = acc
            best_y_test = y_test
            best_y_pred = preds

            if last_model_path is not None and os.path.exists(last_model_path):
                os.remove(last_model_path)

            if last_vocab_path is not None and os.path.exists(last_vocab_path):
                os.remove(last_vocab_path)

            last_model_path = joblib.dump(model, f'model_{max_inst_seq_length}.dat')[0]
            last_vocab_path = joblib.dump(vectorizer.vocabulary_, f'vectorizer_vocab_{max_inst_seq_length}.dat')[0]

    print(classification_report(best_y_test, best_y_pred, target_names=target_names, digits=4))


if __name__ == '__main__':
    TRAIN_DATA = joblib.load('binaries_train.pkl')
    TEST_DATA = joblib.load('binaries_test.pkl')
    main(TRAIN_DATA, TEST_DATA)
