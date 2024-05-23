import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def KNN():
    # Load the dataset
    df = pd.read_csv('./UMAP/mix.csv')
    X = df[["L", "A", "B"]]
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
    knn.fit(X, y)

    # Load outside data
    df_test = pd.read_csv('./UMAP/pixeltest.csv')

    # Extract features and target variable
    X_test_out = df_test[["L", "A", "B"]]
    y_test_out = df_test["pH"]
    y_test_out = label_encoder.fit_transform(y_test_out)


    # Predict on the test data
    y_pred = knn.predict(X_test_out)
    # Calculate accuracy
    accuracy = accuracy_score(y_test_out, y_pred)
    print("Accuracy:", accuracy)

    # df_test['y_pred'] = label_encoder.inverse_transform(y_pred)

    # df_test.to_csv('./UMAP/mixtest_with_predictions.csv', index=False)

def RF():
    # Load the dataset
    df = pd.read_csv('./UMAP/mix.csv')
    X = df[["L", "A", "B"]]
    y = df["pH"]

    # Convert target variable to numeric values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Initialize the Random Forest classifier
    random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest classifier
    random_forest.fit(X, y)

    # Load outside data
    df_test = pd.read_csv('./UMAP/pixeltest.csv')

    # Extract features and target variable
    X_test_out = df_test[["L", "A", "B"]]
    y_test_out = df_test["pH"]
    y_test_out = label_encoder.fit_transform(y_test_out)


    # Predict on the test data
    y_pred = random_forest.predict(X_test_out)
    # Calculate accuracy
    accuracy = accuracy_score(y_test_out, y_pred)
    print("Accuracy:", accuracy)


if __name__ == '__main__':
    KNN()
    RF()