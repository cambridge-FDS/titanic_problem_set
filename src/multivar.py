# TODO: Plot count pairs of "Age Interval" grouped by "Pclass"



# TODO: Plot count pairs of "Age Interval" grouped by "Embarked"



# TODO: Plot count pairs of "Pclass" grouped by "Fare Interval"




def fam_type_size(df1,df2):
    """
    Iterates over df1 and df2, and creates (or updates) a new column called "Family Type" in each df.
    This new column, "Family Type", is set equal to the existing "Family Size" column for each row in df1 and df2.
    """
    for dataset in [df1, df2]:
        dataset["Family Type"] = dataset["Family Size"]



def cat_famsize(df1,df2):
    """
    Categorises each row in df1 and df2 into family types based on the value of "Family Size". 
    The new "Family Type" column assigns labels like "Single", "Small", and "Large" depending on the family size.
    """
    for dataset in [df1, df2]:
        dataset.loc[dataset["Family Size"] == 1, "Family Type"] = "Single"
        dataset.loc[(dataset["Family Size"] > 1) & (dataset["Family Size"] < 5), "Family Type"] = "Small"
        dataset.loc[(dataset["Family Size"] >= 5), "Family Type"] = "Large"



def title_s(df1,df2):
    """
    Iterates over df1 and df2, and creates a new column called "Titles" in each. 
    This new column simply duplicates the values of the existing "Title" column in each dataset.
    """
    for dataset in [df1, df2]:
        dataset["Titles"] = dataset["Title"]




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
    



def xyz(df);
    """
    Groups df by titles and sex and calculates mean
    """
    df[['Titles', 'Sex', 'Survived']].groupby(['Titles', 'Sex'], as_index=False).mean()


