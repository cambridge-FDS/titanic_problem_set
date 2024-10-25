import pandas as pd

def load_data(train_path, test_path):
    """
    Load training and test datasets.

    Parameters:
        train_path (str): Path to the training dataset CSV file.
        test_path (str): Path to the test dataset CSV file.

    Returns:
        pd.DataFrame, pd.DataFrame: Training and test datasets as DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def preprocess_data(df):
    """
    Preprocess the data by filling missing values and encoding categorical features.

    Parameters:
        df (pd.DataFrame): DataFrame to preprocess.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df.drop(['Cabin', 'Ticket', 'Name'], axis=1, inplace=True)
    return df