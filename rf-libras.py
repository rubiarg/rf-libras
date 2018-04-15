import glob as gl
import numpy as np
import pandas as pd
from scipy.io import loadmat

if __name__ == '__main__':
    files = gl.glob('data-libras/*.mat')

    i = 1
    for f in files:
        df = pd.DataFrame(data=loadmat(f).get('pontosSinal'))
        if i > 1:
            X = pd.concat([X, df])
            Y = pd.concat([Y, pd.DataFrame(i * np.ones((10, 1)))])
        else:
            X = df
            Y = pd.DataFrame(i * np.ones((10, 1)))
        i = i + 1

    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.model_selection import RandomizedSearchCV

    rf = RandomForestClassifier(random_state=42)

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
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
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid,
                                   n_iter=100, cv=3, verbose=2,
                                   random_state=42, n_jobs=-1)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.7, random_state=42)
    rf_random.fit(train_x, train_y.values.ravel())

    predictions = rf_random.predict(test_x)

    print("Train Accuracy :: ", accuracy_score(train_y, rf_random.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
    print("Confusion matrix :: \n", confusion_matrix(test_y, predictions))


    def evaluate(model, test_features, test_labels):
        predictions = model.predict(test_features)
        errors = abs(predictions - test_labels)
        mape = 100 * np.mean(errors / test_labels)
        accuracy = 100 - mape
        print('Model Performance')
        print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
        print('Accuracy = {:0.2f}%.'.format(accuracy))

        return accuracy


    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    base_model.fit(train_x, train_y.values.ravel())
    base_accuracy = evaluate(base_model, test_x, test_y)
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, test_x, test_y)
    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))