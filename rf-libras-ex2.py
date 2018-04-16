import glob as gl
import numpy as np
from scipy.io import loadmat
from typing import List


def labelname(file_name):
    label = file_name.replace("data/points/sample", "").lower()
    return label.replace(".mat", "")


class Signal:
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label


def normalize_sig(sig):
    res = np.zeros((np.shape(sig.x)[0], np.shape(sig.x)[1] * 2))
    for idx, x in enumerate(sig.x):
        sig.x[idx] = np.divide((np.transpose(x) - np.mean(x)), np.std(x))
    for idx, y in enumerate(sig.y):
        sig.y[idx] = np.divide((np.transpose(y) - np.mean(y)), np.std(y))

    res[:, ::2] = sig.x
    res[:, 1::2] = sig.y

    return res


if __name__ == "__main__":

    files = gl.glob("data/points/*.mat")  # type: list

    signals = []  # type: List[Signal]

    for f in files:
        data = loadmat(f).get('pontosSinal')
        signals.append(Signal(data[:, ::2], data[:, 1::2], labelname(f)))

    n_signs = len(signals)
    n_recs, n_x = np.shape(signals[0].x)

    signals_norm = []
    signals_label = []
    labels_dict = {}
    i = 0
    for s in signals:
        signals_norm.append(normalize_sig(s))
        signals_label.append([i] * n_recs)
        labels_dict[s.label] = i
        i += 1
    features = np.reshape(signals_norm, (n_signs * n_recs, n_x * 2))
    labels = np.reshape(signals_label, (n_signs * n_recs,))

    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.metrics import classification_report
    from pprint import pprint

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['log2', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 7]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    niter = 30
    for i in range(niter):
        sss = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
        sss.get_n_splits(features, labels)

        for train_index, test_index in sss.split(features, labels):
            train_x, test_x = features[train_index], features[test_index]
            train_y, test_y = labels[train_index], labels[test_index]

        rf = RandomForestClassifier()
        rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100,
                                       cv=3, verbose=1, n_jobs=-1)

        rf_random.fit(train_x, train_y)
        predictions = rf_random.predict(test_x)

        print("Best Params :: "), pprint(rf_random.best_params_)
        print("Train Accuracy :: ", accuracy_score(train_y, rf_random.predict(train_x)))
        print("Test Accuracy  :: ", accuracy_score(test_y, predictions) * 100)
        print("Confusion matrix :: \n", confusion_matrix(test_y, predictions))
        print(classification_report(test_y, predictions, target_names=list(labels_dict)))
