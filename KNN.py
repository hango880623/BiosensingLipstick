import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

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

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("K-Nearest Neighbors:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-score: {f1:.2f}")

def RF(csv_file_path, csv_file_path2, color=["r", "g", "b"]):
    # Load the dataset
    df = pd.read_csv(csv_file_path)
    X = df[color]
    y = df["pH"]

    # Convert target variable to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize the Random Forest classifier
    random_forest = RandomForestClassifier(n_estimators=150, random_state=42)

    # Train the Random Forest classifier
    random_forest.fit(X_train, y_train)

    # Load outside data
    df_test = pd.read_csv(csv_file_path2)

    # Extract features and target variable
    X_test_out = df_test[color]
    y_test_out = df_test["pH"]
    y_test_out = label_encoder.fit_transform(y_test_out)

    # Predict on the test data
    y_pred = random_forest.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Random Forest:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-score: {f1:.2f}")

def SVM(csv_file_path, csv_file_path2, color=["r", "g", "b"]):
    # Load the dataset
    df = pd.read_csv(csv_file_path)
    X = df[color]
    y = df["pH"]

    # Convert target variable to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize the SVM classifier
    svm = SVC(kernel='rbf', gamma=0.5, C=1.0)

    # Train the SVM model
    svm.fit(X_train, y_train)

    # Load outside data
    df_test = pd.read_csv(csv_file_path2)

    # Extract features and target variable
    X_test_out = df_test[color]
    y_test_out = df_test["pH"]
    y_test_out = label_encoder.fit_transform(y_test_out)

    # Predict on the test data
    y_pred = svm.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Support Vector Machine:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-score: {f1:.2f}")

# Predefined LAB values for pH levels
nearest_lab = {
    50: [6.22, 29.84, 12.30],
    60: [20.30, 27.19, 11.98],
    70: [28.22, 35.76, 15.11],
    80: [17.91, 17.94, 3.04]
}

def predict_pH(lab_value):
    # Find the nearest pH value based on Euclidean distance
    nearest_pH = min(nearest_lab.keys(), key=lambda pH: euclidean(lab_value, nearest_lab[pH]))
    return nearest_pH

def Nearest(csv_file_path, csv_file_path2, color = ['L', 'A', 'B']):
    # Load the training dataset (not really used in this method, but keeping it for consistency)
    df_train = pd.read_csv(csv_file_path)
    
    # Load the testing dataset
    df_test = pd.read_csv(csv_file_path2)
    
    # Extract features and target variable
    X_test = df_test[color].values
    y_test = df_test['pH'].values

    # Predict pH values for the test set
    y_pred = np.array([predict_pH(lab) for lab in X_test])

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print("Nearest LAB:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1-score: {f1:.2f}")

if __name__ == '__main__':
    # csv_file_path = './content/Dataset/PaperLOPODiffLAB/Train/median_rgb_values.csv'
    # csv_file_path2 = './content/Dataset/PaperLOPODiffLAB/Test/median_rgb_values.csv'
    csv_file_path = './content/Dataset/Paper/median_rgb_values.csv'
    csv_file_path2 = './content/Dataset/Paper/median_rgb_values.csv'
    Nearest(csv_file_path, csv_file_path2, color=["L", "A", "B"])
    KNN(csv_file_path, csv_file_path2, color=["L", "A", "B"])
    RF(csv_file_path, csv_file_path2, color=["L", "A", "B"])
    SVM(csv_file_path, csv_file_path2, color=["L", "A", "B"])
