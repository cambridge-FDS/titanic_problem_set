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


def pass_sex(df):
    """
    Creates a bar plot showing the number of passengers by sex, 
    broken down by whether the data is from the train or test set.
    """
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
    sns.countplot(x="Sex", data=df, hue="set", palette= color_list)
    plt.grid(color="black", linestyle="-.", linewidth=0.5, axis="y", which="major")
    ax.set_title("Number of passengers / Sex")
    plt.show()  


def survived_sex(df):
    """
    Creates a histogram plot showing the distribution of passengers by sex, 
    segmented by the values in the "Survived" column from the train df. 
    Uses different colors for each unique value of "Survived" 
    (e.g., 0 for those who did not survive and 1 for those who did) 
    and displays a legend indicating the survival status
    """
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(df["Survived"].unique()):
        g = sns.histplot(df.loc[df["Survived"]==h, "Sex"], 
                                    color=color_list[i], 
                                    ax=ax, 
                                    label=h)
    ax.set_title("Number of passengers / Sex")
    g.legend()
    plt.show()



def fam_size(df):
    """
    Used for train and all dfs. Adds a new column called family size which is the
    sum of siblings and parents and adding 1 to account for the individual themselves.
    """
    df["Family Size"] = df["SibSp"] + df["Parch"] + 1
    return df


# Check this code cuz i made it and it's a little shit
# Plot count pairs using all_df for the column "Family Size" and use "Survived" as hue.
def survived_famsize(df):
    """
    Plot count pairs using all_df for the column "Family Size" and use "Survived" as hue.
    """
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(df["Survived"].unique()):
        g = sns.histplot(df.loc[df["Survived"]==h, "Family Size"], 
                                    color=color_list[i], 
                                    ax=ax, 
                                    label=h)
    ax.set_title("Number of passengers / Family Size")
    g.legend()
    plt.show()



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



# Plot count pairs using all_df for the column "Age Interval" and use "Survived" as hue.
def survived_ageint(df):
    """
    Plot count pairs using all_df for the column "Age Interval" and use "Survived" as hue.
    """
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(df["Survived"].unique()):
        g = sns.histplot(df.loc[df["Survived"]==h, "Age Interval"], 
                                    color=color_list[i], 
                                    ax=ax, 
                                    label=h)
    ax.set_title("Number of passengers / Age Interval")
    g.legend()
    plt.show()



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



# Plot count pairs using all_df for the column "Fare Interval"
def survived_ageint(df):
    """
    Plot count pairs using all_df for the column "Fare Interval" and use "Survived" as hue.
    """
    color_list = ["#A5D7E8", "#576CBC", "#19376D", "#0b2447"]
    f, ax = plt.subplots(1, 1, figsize=(8, 4))
    for i, h in enumerate(df["Survived"].unique()):
        g = sns.histplot(df.loc[df["Survived"]==h, "Fare Interval"], 
                                    color=color_list[i], 
                                    ax=ax, 
                                    label=h)
    ax.set_title("Number of passengers / Fare Interval")
    g.legend()
    plt.show()



def sex_class():
    """
    Combines information from both the "Sex" and "Pclass" columns in a specific format
    """
    df["Sex_Pclass"] = df.apply(lambda row: row['Sex'][0].upper() + "_C" + str(row["Pclass"]), axis=1)
    return df



# Plot count pairs using all_df for the column "Fare Interval" and "Fare (grouped by survival)" with "Survived" as hue



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
    
    


def xyz(df):
    """
    Applies the parse_names function to each row in the DataFrame df and assigns the resulting parsed data to new columns
    """
    df[["Family Name", "Title", "Given Name", "Maiden Name"]] = df.apply(lambda row: parse_names(row), axis=1)