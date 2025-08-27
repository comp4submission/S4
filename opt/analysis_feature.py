import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from natsort import natsorted
from xgboost import XGBClassifier, plot_importance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score


def format_display(word: str):
    insts = []
    for i in word.split(' '):
        parts = i.split(',', maxsplit=1)
        opcode = parts[0]

        if opcode == 'indirect_call':
            opcode = '<INDCALL>'
        if opcode == 'internal_call':
            opcode = '<ICALL>'
        if opcode == 'external_call':
            opcode = '<ECALL>'

        operands = None

        if len(parts) > 1:
            operands = parts[1].upper()

        new = f'{opcode} {operands}' if operands else opcode
        insts.append(new)
    return ' | '.join(insts)


def main():
    model_file = 'model_40000.dat'
    model: OneVsRestClassifier = joblib.load(model_file)
    vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None,
                                 vocabulary=joblib.load('vectorizer_vocab_40000.dat'))

    opt_to_id = joblib.load('opt_to_id.dat')
    id_to_opt: dict[int, str] = {v: k for k, v in opt_to_id.items()}

    top_n = 3

    opt_to_top_n_features = {}

    for i, e in enumerate(model.estimators_):
        opt = id_to_opt[i]
        feature_to_importance = dict(zip(vectorizer.get_feature_names_out(), e.feature_importances_))
        feature_importance = sorted(feature_to_importance.items(), key=lambda x: x[1], reverse=True)

        top_n_features = [feat[0] for feat in feature_importance[:top_n]]
        opt_to_top_n_features[opt] = list(map(lambda x: format_display(x), top_n_features))

    for opt in natsorted(opt_to_top_n_features.keys()):
        print(f'{opt}')
        for feat in opt_to_top_n_features[opt]:
            print(feat)
        print()


if __name__ == '__main__':
    main()
