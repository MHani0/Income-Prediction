import pandas as pd
import numpy as np


def preprocessing(df):

    print("\nPre-processing Statistics:\n")
    # remove whitespaces from string columns
    for col in df.columns:
        if df[col].dtype == 'object':  # check if the column contains strings
            df[col] = df[col].str.strip()
        # illegal format check
        elif col in ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            df.loc[df[col] < 0, col] = 0

    # replacing "?" with NaN
    df.replace('?', np.nan, inplace=True)

    # NaN statistics
    nulls = df.isna().sum()
    print("\nNaN Stats:\n\n")

    for col, index in zip(nulls, df.columns):
        nan_percentage = round(col * 100 / len(df), 2)
        print(index + "\n" + str(col) + "\n" + str(nan_percentage) + "%\n")

## Workable for later
    # fill null with mode
    df['workclass'].fillna(df['workclass'].mode()[0], inplace=True)
    df['native-country'].fillna(df['native-country'].mode()[0], inplace=True)

    # drop null
    df.dropna(subset=['occupation'], inplace=True)
##

    # mapping categorical values to numerals
    for col in df.columns:
        if col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            df[col] = df[col].astype(int)
            continue
        df[col], _ = pd.factorize(df[col])

    # display number of duplicated rows and null
    duplicates = df.duplicated().sum()
    print("\nDuplicates found: " + str(duplicates) + "\n")

    df.drop_duplicates(inplace=True)

    df.reset_index(drop=True, inplace=True)

    print(df)


# read csv file
train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv('data/test_data.csv')

preprocessing(train_df)
preprocessing(test_df)