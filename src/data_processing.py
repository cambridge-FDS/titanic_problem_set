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

def add_age_interval(df):

    df["Age Interval"] = 0.0
    
    # Categorize ages into different intervals
    df.loc[df['Age'] <= 16, 'Age Interval'] = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[df['Age'] > 64, 'Age Interval'] = 4
    
    return df

def add_fare_interval(df):
    """
    Categorize the 'Fare' column into intervals and create a new 'Fare Interval' column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Fare' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'Fare Interval' column.
    """
    # Initialize the 'Fare Interval' column
    df['Fare Interval'] = 0.0
    
    # Categorize fares into different intervals
    df.loc[df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval'] = 2
    df.loc[df['Fare'] > 31, 'Fare Interval'] = 3
    
    return df

def add_sex_pclass_feature(df):
    """
    Create a new 'Sex_Pclass' feature by combining the first letter of 'Sex' 
    and 'Pclass' to indicate passenger class.

    Parameters:
        df (pd.DataFrame): DataFrame containing 'Sex' and 'Pclass' columns.

    Returns:
        pd.DataFrame: DataFrame with an added 'Sex_Pclass' column.
    """
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df

def add_family_type(df):
    """
    Categorize 'Family Size' into 'Family Type' and add it as a new column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Family Size' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'Family Type' column.
    """
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[df["Family Size"] >= 5, "Family Type"] = "Large"
    return df


def unify_titles(df):
    """
    Unify titles in the 'Titles' column by replacing variations of common titles and rare titles.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Titles' column.

    Returns:
        pd.DataFrame: DataFrame with unified 'Titles' values.
    """
    # Unify 'Miss'
    df['Titles'] = df['Titles'].replace(['Mlle.', 'Ms.'], 'Miss.')
    
    # Unify 'Mrs'
    df['Titles'] = df['Titles'].replace('Mme.', 'Mrs.')
    
    # Unify rare titles
    df['Titles'] = df['Titles'].replace(
        ['Lady.', 'the Countess.', 'Capt.', 'Col.', 'Don.', 'Dr.', 
         'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
    
    return df

def add_family_type(df):
    """
    Categorize 'Family Size' into 'Family Type' and add it as a new column.

    Parameters:
        df (pd.DataFrame): DataFrame containing the 'Family Size' column.

    Returns:
        pd.DataFrame: DataFrame with an added 'Family Type' column.
    """
    df.loc[df["Family Size"] == 1, "Family Type"] = "Single"
    df.loc[(df["Family Size"] > 1) & (df["Family Size"] < 5), "Family Type"] = "Small"
    df.loc[df["Family Size"] >= 5, "Family Type"] = "Large"
    return df
