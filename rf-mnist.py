import mnist
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

if __name__ == '__main__':
    train_images = mnist.train_images()
    train_x = train_images.reshape((train_images.shape[0], train_images.shape[1]*train_images.shape[2]))
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_x = test_images.reshape((test_images.shape[0], test_images.shape[1]*test_images.shape[2]))
    test_labels = mnist.test_labels()

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

    rf_random.fit(train_x, train_labels)

    predictions = rf_random.predict(test_labels)

    print("Train Accuracy :: ", accuracy_score(train_labels, rf_random.predict(train_x)))
    print("Test Accuracy  :: ", accuracy_score(test_labels, predictions))
    print("Confusion matrix :: \n", confusion_matrix(test_labels, predictions))

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
    base_model.fit(train_x, train_labels)
    base_accuracy = evaluate(base_model, test_y, test_labels)
    best_random = rf_random.best_estimator_
    random_accuracy = evaluate(best_random, test_y, test_labels)
    print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))