import pandas as pd
import numpy as np

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



def miss_unique(df):
    """
    For each column in a df, calculate the total number of missing values and the number of unique values
    """
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in df.columns:
        unique = df[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    df_missunique= np.transpose(tt)
    return df_missunique