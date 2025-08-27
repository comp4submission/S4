import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

from natsort import natsorted
from xgboost import XGBClassifier, plot_importance
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import CountVectorizer


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
    model_file = 'model_32000.dat'
    model: OneVsRestClassifier = joblib.load(model_file)
    vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=lambda x: x.split(' '), token_pattern=None,
                                 vocabulary=joblib.load('vectorizer_vocab.dat'))

    compiler_to_id = joblib.load('compiler_to_id.dat')
    id_to_compiler: dict[int, str] = {v: k for k, v in compiler_to_id.items()}

    top_n = 3

    compiler_to_top_n_features = {}

    for i, e in enumerate(model.estimators_):
        compiler = id_to_compiler[i]
        feature_to_importance = dict(zip(vectorizer.get_feature_names_out(), e.feature_importances_))
        feature_importance = sorted(feature_to_importance.items(), key=lambda x: x[1], reverse=True)

        top_n_features = [feat[0] for feat in feature_importance[:top_n]]
        compiler_to_top_n_features[compiler] = list(map(lambda x: format_display(x), top_n_features))

    for compiler in natsorted(compiler_to_top_n_features.keys()):
        print(f'{compiler}')
        for feat in compiler_to_top_n_features[compiler]:
            print(feat)
        print()


if __name__ == '__main__':
    main()
