import pandas as pd
import numpy as np

def calculate_missing_data(train_df):
    """
    Calculate the total and percentage of missing data for each column in the DataFrame,
    along with the data types of the columns.

    Parameters:
    train_df (pd.DataFrame): The input DataFrame to calculate missing data for.

    Returns:4
    pd.DataFrame: A DataFrame with the total missing values, percentage of missing values,
                  and the data type of each column.
    """
    total = train_df.isnull().sum()
    percent = (train_df.isnull().sum() / train_df.isnull().count() * 100)
    
    # Concatenate total and percentage of missing values
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Add the data types
    types = [str(train_df[col].dtype) for col in train_df.columns]
    tt['Types'] = types
    
    # Transpose the result
    df_missing_train = np.transpose(tt)
    
    return df_missing_train


def missing_data_summary(df):
    """
    Generate a summary of missing data for a given DataFrame.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze for missing data.
    
    Returns:
    pd.DataFrame: A transposed DataFrame showing total missing values, 
                  percentage of missing values, and data types for each column.
    """
    total = df.isnull().sum()
    percent = (df.isnull().sum() / df.isnull().count()) * 100
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Collect the data types of each column
    types = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        types.append(dtype)
    
    # Add the data types to the summary
    tt['Types'] = types
    
    # Transpose the DataFrame for readability
    df_missing = np.transpose(tt)
    
    return df_missing

# Example usage:
# df_missing_test = missing_data_summary(test_df)

def frequent_item_summary(df):
    """
    Generate a summary DataFrame that includes total counts, the most frequent 
    item in each column, its frequency, and its percentage of the total.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    pd.DataFrame: A transposed DataFrame with the following columns:
        - Total: Total count of non-null items per column.
        - Most frequent item: The most frequent item in each column.
        - Frequency: Frequency of the most frequent item.
        - Percent from total: The percentage of the most frequent item's frequency 
          relative to the total non-null items in that column.
    """
    # Total count of non-null items per column
    total = df.count()
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    items = []
    vals = []
    
    # Iterate over each column to find the most frequent item and its frequency
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]  # Most frequent item
            val = df[col].value_counts().values[0]  # Frequency of the most frequent item
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(ex)
            items.append(0)  # If an error occurs, append 0
            vals.append(0)
            continue
    
    # Add 'Most frequent item' and 'Frequency' columns to the DataFrame
    tt['Most frequent item'] = items
    tt['Frequency'] = vals
    
    # Calculate the percentage of the most frequent item from the total
    tt['Percent from total'] = np.round(np.array(vals) / total * 100, 3)
    
    # Transpose for readability
    df_summary = np.transpose(tt)
    
    return df_summary

# Example usage:
# summary_df = frequent_item_summary(train_df)

def generate_frequent_item_summary(df):
    """
    Generate a summary DataFrame that includes total counts, the most frequent 
    item in each column, its frequency, and its percentage of the total.
    
    Parameters:
    df (pd.DataFrame): The DataFrame to analyze.
    
    Returns:
    pd.DataFrame: A transposed DataFrame with the following columns:
        - Total: Total count of non-null items per column.
        - Most frequent item: The most frequent item in each column.
        - Frequency: Frequency of the most frequent item.
        - Percent from total: The percentage of the most frequent item's frequency 
          relative to the total non-null items in that column.
    """
    # Total count of non-null items per column
    total = df.count()
    tt = pd.DataFrame(total, columns=['Total'])
    
    items = []
    vals = []
    
    # Iterate over each column to find the most frequent item and its frequency
    for col in df.columns:
        try:
            itm = df[col].value_counts().index[0]  # Most frequent item
            val = df[col].value_counts().values[0]  # Frequency of the most frequent item
            items.append(itm)
            vals.append(val)
        except Exception as ex:
            print(f"Error processing column {col}: {ex}")
            items.append(0)  # If an error occurs, append 0
            vals.append(0)
            continue
    
    # Add 'Most frequent item' and 'Frequency' columns to the DataFrame
    tt['Most frequent item'] = items
    tt['Frequency'] = vals
    
    # Calculate the percentage of the most frequent item from the total
    tt['Percent from total'] = np.round(np.array(vals) / total * 100, 3)
    
    # Transpose for readability
    df_summary = np.transpose(tt)
    
    return df_summary

# Example usage:
# test_summary_df = generate_frequent_item_summary(test_df)

def summarize_dataframe(df):
    """
    Compute the total count and number of unique values for each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: Transposed DataFrame containing total counts and unique values per column.
    """
    # Compute the total count for each column
    total = df.count()
    
    # Create a DataFrame to hold the results
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    # Compute the number of unique values for each column
    uniques = [df[col].nunique() for col in df.columns]
    tt['Uniques'] = uniques
    
    # Transpose the DataFrame to get the desired format
    return np.transpose(tt)

# Example usage:
# result = summarize_dataframe(train_df)
# print(result)

def summarize_test_dataframe(df):
    """
    Compute the total count and number of unique values for each column in the test DataFrame.

    Parameters:
    df (pd.DataFrame): The input test DataFrame.

    Returns:
    pd.DataFrame: Transposed DataFrame containing total counts and unique values per column.
    """
    # Compute the total count for each column
    total = df.count()
    
    # Create a DataFrame to hold the results
    tt = pd.DataFrame(total)
    tt.columns = ['Total']
    
    # Compute the number of unique values for each column
    uniques = [df[col].nunique() for col in df.columns]
    tt['Uniques'] = uniques
    
    # Transpose the DataFrame to get the desired format
    return np.transpose(tt)

# Example usage:
# result = summarize_test_dataframe(test_df)
# print(result)
