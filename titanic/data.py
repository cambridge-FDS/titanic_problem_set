import pandas as pd
import numpy as np
from pathlib import Path

train_path = Path(__file__).parent/"data"/"train.csv"
test_path = Path(__file__).parent/"data"/"tests.csv"

def load(path):
    df = pd.read_csv(path)
    return df

def miss_total(df):
    """ 
    Creates a dataframe displaying the total number and percentage of missing values for each column, along with the datatype.
    """
    total = df.isnull().sum()
    percent = (df.isnull().sum()/df.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent']) #Creates a new df which concatenates total and percentage missing
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    tt['Types'] = types                                                                     
    df_missing = np.transpose(tt) #Transposes tt for easier viewing
    return df_missing



def miss_summ(df):
    """
    This code analyzes each column in a df to identify the total number of non-missing values, 
    the most frequent value (mode), its frequency, and the percentage of this frequency relative to the total 
    non-missing values in each column.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]
            val = df[col].value_counts().values[0]
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)
            vals.append(0)
            continue
    tt['Most frequent item'] = items
    tt['Frequence'] = vals
    tt['Percent from total'] = np.round(vals / total * 100, 3)
    summ_df= np.transpose(tt)
    return summ_df


def unique(df):
    """
    Calculates the number of unique values for each column in a df. 
    Stores the total non-missing values and the count of unique values in each column within a new df. 
    Finally, it transposes the resulting DataFrame (tt) for easier viewing.
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    df_unique=np.transpose(tt)
    return df_unique

"""
We show here two graphs in paralel:
* distribution of class values, split per Survived value
* comparison of class values, in train and test data


Let's first aggregate train and test data into one single dataframe, `all_df`.
"""
import pandas as pd
import matplotlib as plt
import seaborn as sns


def concat(df1,df2):
    """
    Combines 2 datasets. Creates a new coloumn called set which labels rows from test 
    values df2 as test because they don't have values for survived.
    """
    all_df = pd.concat([df1, df2], axis=0)
    all_df["set"] = "train"
    all_df.loc[all_df.Survived.isna(), "set"] = "test"
    return all_df

def no_duplicate(df):
    print(df.columns.duplicated())
    print(df.index.duplicated())
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.reset_index(drop=True)
    return df

def fam_size(df):
    """
    Used for train and all dfs. Adds a new column called family size which is the
    sum of siblings and parents and adding 1 to account for the individual themselves.
    """
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df

def age_int(df):
    """
    Creates an "Age Interval" column in df, which groups passengers’ ages into intervals or age brackets.
    """
    df["Age Interval"] = 0.0
    df.loc[ df['Age'] <= 16, 'Age Interval']  = 0
    df.loc[(df['Age'] > 16) & (df['Age'] <= 32), 'Age Interval'] = 1
    df.loc[(df['Age'] > 32) & (df['Age'] <= 48), 'Age Interval'] = 2
    df.loc[(df['Age'] > 48) & (df['Age'] <= 64), 'Age Interval'] = 3
    df.loc[ df['Age'] > 64, 'Age Interval'] = 4
    return df

def fare_int(df):
    """
    Creates an "Age Interval" column in df, which groups passengers’ fares into intervals.
    """
    df['Fare Interval'] = 0.0
    df.loc[ df['Fare'] <= 7.91, 'Fare Interval'] = 0
    df.loc[(df['Fare'] > 7.91) & (df['Fare'] <= 14.454), 'Fare Interval'] = 1
    df.loc[(df['Fare'] > 14.454) & (df['Fare'] <= 31), 'Fare Interval']   = 2
    df.loc[ df['Fare'] > 31, 'Fare Interval'] = 3
    return df

def sex_class(df):
    """
    Combines information from both the "Sex" and "Pclass" columns in a specific format
    """
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df

def parse_names(row):
    """
    Takes a row from a df and attempts to parse the "Name" column into separate components like family_name, title, given_name, and maiden_name
    """
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

def titles(df1,df2):
    """
    Standardising titles across df1 and df2
    """
    for dataset in [df1, df2]:
        #unify `Miss`
        dataset['Titles'] = dataset['Titles'].replace('Mlle.', 'Miss.')
        dataset['Titles'] = dataset['Titles'].replace('Ms.', 'Miss.')
        #unify `Mrs`
        dataset['Titles'] = dataset['Titles'].replace('Mme.', 'Mrs.')
        # unify Rare
        dataset['Titles'] = dataset['Titles'].replace(['Lady.', 'the Countess.','Capt.', 'Col.',\
        'Don.', 'Dr.', 'Major.', 'Rev.', 'Sir.', 'Jonkheer.', 'Dona.'], 'Rare')
    return dataset

def analys_table_survived_by_x_y(df, x, y):
    """
    Create a table showing the number of survived and not survived passengers for each unique value of x and y.
    """
    return df[[x, y, "Survived"]].groupby([x, y], as_index=False).mean()

def feat_sex_numeric(dataset):
    mapped_sex = dataset["Sex"].map({"female": 1, "male": 0})
    if mapped_sex.isna().any():
        print("Warning: NaN values found after mapping!")
        print(mapped_sex.isna().sum(), "NaN values")
    dataset["Sex"] = mapped_sex.astype(int)
    return dataset