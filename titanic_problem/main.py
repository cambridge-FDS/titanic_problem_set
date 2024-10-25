from preliminary_inspection import load_data, find_missing_data, find_most_frequent, find_uniques

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"

def main():
    # Load train and test data
    train_df = load_data(TRAIN_PATH)
    test_df = load_data(TEST_PATH)

    # Sanity check
    train_df.head()
    test_df.head()
    train_df.info()
    test_df.info()
    train_df.describe()
    test_df.describe()

    # Preliminary inspection
    df_missing_train = find_missing_data(train_df)
    df_missing_test = find_missing_data(test_df)
    df_most_frequent_train = find_most_frequent(train_df)
    df_most_frequent_test = find_most_frequent(test_df)
    df_train_unique = find_uniques(train_df)
    df_test_unique = find_uniques(test_df)


if __name__ == "__main__":
    main()