from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier



def tt_split_valid(df, VALID_SIZE):
    """
    Splits the df into two separate dfs, train and valid, creating a training set and a validation set for model training and evaluation.
    """
    train, valid = train_test_split(df, test_size=VALID_SIZE, random_state=42, shuffle=True)
    return train, valid



predictors = ["Sex", "Pclass"]
target = 'Survived'



def sep_trainvalid(df1,df2):
    """
    Separates the training and validation data into features (predictors) and target (the label or outcome to predict)
    """
    train_X = df1[predictors]
    train_Y = df1[target].values
    valid_X = df2[predictors]
    valid_Y = df2[target].values
    return train_X, train_Y, valid_X, valid_Y



def asd(df1,df2):
    """
    Initialises the classifier and trains the model using the train data
    """
    clf = RandomForestClassifier(n_jobs=-1, 
                                random_state=42,
                                criterion="gini",
                                n_estimators=100,
                                verbose=False)
    clf.fit(df1, df2)
