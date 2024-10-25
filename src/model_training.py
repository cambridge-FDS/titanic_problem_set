from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def split_data(df):
    """
    Split the dataset into features and target variable, and then into training and validation sets.

    Parameters:
        df (pd.DataFrame): DataFrame to split.

    Returns:
        tuple: Features and target variable for both training and validation sets.
    """
    X = df.drop(['Survived'], axis=1)
    y = df['Survived']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """
    Train a RandomForest model.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target variable.

    Returns:
        RandomForestClassifier: Trained RandomForest model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluate the trained model.

    Parameters:
        model (RandomForestClassifier): Trained model.
        X_val (pd.DataFrame): Validation features.
        y_val (pd.Series): Validation target variable.

    Returns:
        float: Accuracy of the model on the validation set.
    """
    y_pred = model.predict(X_val)
    return accuracy_score(y_val, y_pred)

