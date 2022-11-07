
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics, svm


import pandas as pd
import numpy as np

def load_data(filepath):
    return pd.read_csv(filepath)


def generate_patterns_1():
    patterns = []
    for i in range(1, 10):
        patterns.append([f"F{i}"])
    print("Done 1")
    return set(tuple(i) for i in patterns)

def generate_patterns_2():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            patterns.append([f"F{i}", f"F{j}"])
    print("Done 2")
    return set(tuple(i) for i in patterns)

def generate_patterns_3():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                patterns.append([f"F{i}", f"F{j}", f"F{k}"])

    print("Done 3")
    return set(tuple(i) for i in patterns)

def generate_patterns_4():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                for l in range(1, 10):
                    patterns.append([f"F{i}", f"F{j}", f"F{k}", f"F{l}"])
    print("Done 4")
    return set(tuple(i) for i in patterns)

def generate_patterns_5():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                for l in range(1, 10):
                    for m in range(1, 10):
                        patterns.append([f"F{i}", f"F{j}", f"F{k}", f"F{l}", f"F{m}"])
    print("Done 5")
    return set(tuple(i) for i in patterns)

def generate_patterns_6():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                for l in range(1, 10):
                    for m in range(1, 10):
                        for n in range(1, 10):
                            patterns.append([f"F{i}", f"F{j}", f"F{k}", f"F{l}", f"F{m}", f"F{n}"])
    print("Done 6")
    return set(tuple(i) for i in patterns)


def generate_patterns_7():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                for l in range(1, 10):
                    for m in range(1, 10):
                        for n in range(1, 10):
                            for o in range(1, 10):
                                patterns.append([f"F{i}", f"F{j}", f"F{k}", f"F{l}", f"F{m}", f"F{n}", f"F{o}"])
    print("Done 7")
    return set(tuple(i) for i in patterns)

def generate_patterns_8():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                for l in range(1, 10):
                    for m in range(1, 10):
                        for n in range(1, 10):
                            for o in range(1, 10):
                                for p in range(1, 10):
                                    patterns.append([f"F{i}", f"F{j}", f"F{k}", f"F{l}", f"F{m}", f"F{n}", f"F{o}", f"F{p}"])
    print("Done 8")
    return set(tuple(i) for i in patterns)


def generate_patterns_9():
    patterns = []
    for i in range(1, 10):
        for j in range(1, 10):
            for k in range(1, 10):
                for l in range(1, 10):
                    for m in range(1, 10):
                        for n in range(1, 10):
                            for o in range(1, 10):
                                for p in range(1, 10):
                                    for q in range(1, 10):
                                        patterns.append([f"F{i}", f"F{j}", f"F{k}", f"F{l}", f"F{m}", f"F{n}", f"F{o}", f"F{p}", f"F{q}"])
    print("Done 9")
    return set(tuple(i) for i in patterns)

def generate_patterns():
    patterns = [
    generate_patterns_1(),
    generate_patterns_2(),
    generate_patterns_3(),
    generate_patterns_4(),
    generate_patterns_5(),
    generate_patterns_6(),
    generate_patterns_7(),
    generate_patterns_8(),
    generate_patterns_9()
    ]

    return set(tuple(i) for i in patterns)


def prepare_dataset(dataset):
    X = dataset.iloc[:, 1:-1].to_numpy()
    Y = dataset.iloc[:, -1:].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    return X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    dataset = load_data("breast-cancer-wisconsin.data")

    model_knn = KNeighborsClassifier(n_neighbors=3)
    model_svm = svm.SVC(kernel='linear')

    X_train, X_test, Y_train, Y_test = prepare_dataset(dataset=dataset)

    model_knn.fit(X_train, Y_train.ravel())
    model_svm.fit(X_train, Y_train.ravel())

    Y_knn_pred = model_knn.predict(X_test)
    Y_svm_pred = model_svm.predict(X_test)

    print("Accuracy of KNN: ", metrics.accuracy_score(Y_test, Y_knn_pred))
    print("Accuracy of SVM: ", metrics.accuracy_score(Y_test, Y_svm_pred))
    patterns = generate_patterns()


    for pattern in patterns:
        for feature_set in pattern:
            model = svm.SVC(kernel="linear")
            X = []
            for feature in feature_set:
                temp = dataset.loc[:, feature]
                X.append(temp)
            X = np.array(X).reshape(683, -1)
            X_train, X_test, Y_train, Y_test = train_test_split(X, dataset.iloc[:, -1:].to_numpy(), test_size=0.3)
            model.fit(X_train, Y_train.ravel())
            Y_pred = model.predict(X_test)
            print(f"Accuracy of {feature_set}: {metrics.accuracy_score(Y_test, Y_pred)}")            
