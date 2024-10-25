import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def find_missing_data(dfr: pd.DataFrame) -> pd.DataFrame:
    df_missing = pd.DataFrame()
    total = dfr.isnull().sum()
    percent = (dfr.isnull().sum()/dfr.isnull().count()*100)
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    types = []
    for col in dfr.columns:
        dtype = str(dfr[col].dtype)
        types.append(dtype)
    tt['Types'] = types
    df_missing = np.transpose(tt)
    return df_missing


def find_most_frequent(dfr: pd.DataFrame) -> pd.DataFrame:
    total = dfr.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    items = []
    vals = []
    for col in dfr.columns:
        try:
            itm = dfr[col].value_counts().index[0]
            val = dfr[col].value_counts().values[0]
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
    return np.transpose(tt)


def find_uniques(dfr: pd.DataFrame) -> pd.DataFrame:
    total = dfr.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    uniques = []
    for col in dfr.columns:
        unique = dfr[col].nunique()
        uniques.append(unique)
    tt['Uniques'] = uniques
    return np.transpose(tt)