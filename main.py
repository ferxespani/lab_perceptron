from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def predict(X, y):
    scores_first_class = model_first_class.decision_function(X)
    scores_second_class = model_second_class.decision_function(X)
    scores_third_class = model_third_class.decision_function(X)

    Y = []
    for i in range(scores_first_class.shape[0]):
        argmax = np.argmax([scores_first_class[i],
                            scores_second_class[i],
                            scores_third_class[i]])
        Y.append(argmax)

    predictions = []
    for i in range(len(Y)):
        if Y[i] == y[i]:
            predictions.append(1)
        else:
            predictions.append(0)

    return predictions


def calculate_accuracy(predictions):
    correct_predictions = 0
    for i in range(len(predictions)):
        if predictions[i] == 1:
            correct_predictions += 1

    return correct_predictions / len(predictions) * 100


if __name__ == '__main__':
    dataset = load_wine()
    features = dataset.data
    target = dataset.target

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.20, random_state=42)

    normalized_range = MinMaxScaler(feature_range=(-1, 1))
    #normalized_range = StandardScaler()
    X_train = normalized_range.fit_transform(X_train)
    X_test = normalized_range.fit_transform(X_test)

    target_first_class = (y_train == 0).astype(int)
    model_first_class = Perceptron()
    model_first_class.fit(X_train, target_first_class)

    target_second_class = (y_train == 1).astype(int)
    model_second_class = Perceptron()
    model_second_class.fit(X_train, target_second_class)

    target_third_class = (y_train == 2).astype(int)
    model_third_class = Perceptron()
    model_third_class.fit(X_train, target_third_class)

    predicts = predict(X_test, y_test)
    print(f"Accuracy: {calculate_accuracy(predicts)}")

    print(f"Size of the dataset: {dataset.data.shape}")

    print("Feature names:")
    print(dataset.feature_names)

    print(f"Number of classes: {dataset.target_names.shape[0]}")

    df = pd.DataFrame(features, columns=dataset.feature_names)
    df["class"] = target

    for k in range(dataset.target_names.shape[0]):
        t = df[df["class"] == k]
        mean = t.mean(axis=0)
        min_value = t.min(axis=0)
        max_value = t.max(axis=0)

        print(f"Mean value for the {k} class: \n {mean}")
        print(f"Min value for the {k} class:  \n {min_value}")
        print(f"Max value for the {k} class:  \n {max_value}")

