import pandas as pd
import numpy as np
import seaborn as sns

def concat_train_test(train_df, test_df):
    """
    Combine train and test data into a single dataframe
    
    Args:
    train_df (pd.DataFrame):  training data
    test_df (pd.DataFrame):  test data
    
    Returns:
    all_df (pd.DataFrame): combined dataframe of train and test data
    """
    all_df = pd.concat([train_df, test_df], axis=0)
    #set identifi
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df


def set_family_size(df):
    """ Set Family Size column based on SinSp and Parch columns
    
    Args:
    df (pd.DataFrame):  dataframe containing SibSp and Parch columns
    
    Returns:
    df (pd.DataFrame): dataframe with Family Size column added
    """
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df



def set_age_interval(df):
    """
    Set Age Interval column based on Age column
    
    Args:
    df (pd.DataFrame):  dataframe containing Age columns
    
    Returns: 
    df (pd.DataFrame): dataframe with Age Interval column added
    """

    df["Age Interval"] = 0.0
    df.loc[ df['Age'] <= 16, 'Age Interval']  = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[ df['Age'] > 64, 'Age Interval'] = 4

    return df


def set_fare_interval(df):
    """
    Set Fare Interval column based on Fare column
    
    Args:
    df (pd.DataFrame):  dataframe containing Fare columns
    
    Returns: 
    df (pd.DataFrame): dataframe with Fare Interval column added
    """
    df['Fare Interval'] = 0.0
    df.loc[ df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval']   = 2
    df.loc[ df['Fare'] > 31, 'Fare Interval'] = 3
    return df

def create_sex_Pclass(df):
    """
    Create a composed feature: Pclass + Sex
    
    Args:
    df (pd.DataFrame):  dataframe containing Pclass and Sex columns
    
    Returns: 
    df (pd.DataFrame): dataframe with Sex_Pclass column added
    """

    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df


def parse_names(row):
    try:
        text = row["Name"]
        split_text = text.split(",")
        family_name = split_text[0]
        next_text = split_text[1]
        split_text = next_text.split(".")
        title = (split_text[0] + ".").lstrip().rstrip()
        next_text = split_text[1]
        if "(" in next_text:
            split_text = next_text.split("(")
            given_name = split_text[0]
            maiden_name = split_text[1].rstrip(")")
            return pd.Series([family_name, title, given_name, maiden_name])
        else:
            given_name = next_text
            return pd.Series([family_name, title, given_name, None])
    except Exception as ex:
        print(f"Exception: {ex}")
    
    


def process_name(df):
    """
    Extract information from Name column and create Family Name, Title, Given Name, Maiden Name columns
    
    Args:
    df (pd.DataFrame):  dataframe containing Name column
    
    Returns: 
    df (pd.DataFrame): dataframe with Family Name, Title, Given Name, Maiden Name columns added
    """

    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)
    return df
    
