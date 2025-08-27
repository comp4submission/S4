import time

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


def single_eval(model, X_train, X_test, y_train, y_test):
    print('Start Single Evaluation')
    start_time = time.time()
    model.fit(X_train, y_train)
    end_time = time.time()
    # make predictions
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds), preds


def cross_val(model, X, y, cv=5):
    print('Start Cross Validation')
    start_time = time.time()
    scores = cross_val_score(model, X, y, cv=cv)
    end_time = time.time()

    avg_acc = sum(scores) / len(scores)
    time_cost = end_time - start_time
    return scores, time_cost
