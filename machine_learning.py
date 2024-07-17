import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from scipy.spatial.distance import euclidean
import numpy as np


def KNN(csv_file_path, csv_file_path2, color=["r", "g", "b"]):
    # Load the dataset
    df = pd.read_csv(csv_file_path)
    X = df[color]
    y = df["pH"]

    # Convert target variable to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # Initialize KNN classifier
    k = 3  # Number of neighbors
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Load outside data
    df_test = pd.read_csv(csv_file_path2)

    # Extract features and target variable
    X_test_out = df_test[color]
    y_test_out = df_test["pH"]
    y_test_out = label_encoder.fit_transform(y_test_out)

    # Predict on the test data
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))


def RF(csv_file_path, csv_file_path2, color=["r", "g", "b"]):
    df = pd.read_csv(csv_file_path)
    X = df[color]
    y = df["pH"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # parameter grids
    param_grid = {
        'n_estimators': [25, 50, 100, 150],
        'max_features': ['sqrt', 'log2', None],
        'max_depth': [3, 6, 9],
        'max_leaf_nodes': [3, 6, 9],
    }
    # # Initialize GridSearchCV with 5-fold cross-validation
    # grid_search = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=5)
    # grid_search.fit(X_train, y_train)
    # print(grid_search.best_estimator_)

    df_test = pd.read_csv(csv_file_path2)

    X_test_out = df_test[color]
    y_test_out = df_test["pH"]
    y_test_out = label_encoder.fit_transform(y_test_out)

    model_grid = RandomForestClassifier(max_depth=9,
                                        max_features="log2",
                                        max_leaf_nodes=9,
                                        n_estimators=50)
    model = RandomForestClassifier()

    model_grid.fit(X_train, y_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_pred_grid = model_grid.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(classification_report(y_test, y_pred_grid))


def SVM(csv_file_path, csv_file_path2, color=["r", "g", "b"]):
    df = pd.read_csv(csv_file_path)
    X = df[color]
    y = df["pH"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42)

    # # Define paramerter range
    # param_grid = {'C': [0.1, 1, 10, 100, 1000],
    #           'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    #           'kernel': ['rbf']}
    # grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3)

    # grid.fit(X_train, y_train)

    # print(grid.best_params_)
    # print(grid.best_estimator_)

    # Initialize the SVM classifier
    svm = SVC()
    svm_grid = SVC(kernel='rbf', gamma=0.001, C=1000)

    # Train the SVM model
    svm.fit(X_train, y_train)
    svm_grid.fit(X_train, y_train)

    # Load outside data
    df_test = pd.read_csv(csv_file_path2)

    # Extract features and target variable
    X_test_out = df_test[color]
    y_test_out = df_test["pH"]
    y_test_out = label_encoder.fit_transform(y_test_out)

    # Predict on the test data
    y_pred = svm.predict(X_test)
    y_pred_grid = svm_grid.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(classification_report(y_test, y_pred_grid))


def predict_pH(lab_value):
    # Predefined LAB values for pH levels
    nearest_lab = {
        50: [6.22, 29.84, 12.30],
        60: [20.30, 27.19, 11.98],
        70: [28.22, 35.76, 15.11],
        80: [17.91, 17.94, 3.04]
    }
    # Find the nearest pH value based on Euclidean distance
    nearest_pH = min(nearest_lab.keys(), key=lambda pH: euclidean(
        lab_value, nearest_lab[pH]))
    return nearest_pH


def Nearest(csv_file_path, csv_file_path2, color=['L', 'A', 'B']):
    df_train = pd.read_csv(csv_file_path)
    df_test = pd.read_csv(csv_file_path2)

    X_test = df_test[color].values
    y_test = df_test['pH'].values

    y_pred = np.array([predict_pH(lab) for lab in X_test])
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    # csv_file_path = './content/Dataset/PaperLOPODiffLAB/Train/median_rgb_values.csv'
    # csv_file_path2 = './content/Dataset/PaperLOPODiffLAB/Test/median_rgb_values.csv'
    csv_file_path = './content/Dataset/Paper512/median_rgb_values.csv'
    csv_file_path2 = './content/Dataset/Paper512/median_rgb_values.csv'
    # Nearest(csv_file_path, csv_file_path2, color=["L", "A", "B"])
    # KNN(csv_file_path, csv_file_path2, color=["L", "A", "B"])
    SVM(csv_file_path, csv_file_path2, color=["L", "A", "B"])
    # SVM(csv_file_path, csv_file_path2, color=["L", "A", "B"])
