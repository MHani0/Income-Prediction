import pandas as pd
import numpy as np
# read csv file
cols = ['Age', 'Work Class', 'Final Weight', 'Type of Education', 'Years of Education', 'Marital Status',
        'Occupation', 'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss', 'Hours/Week', 'Native Country',
        'Income']

train_df = pd.read_csv("data/train_data.csv", names=cols)
test_df = pd.read_csv('data/test_data.csv')

print("\nPre-processing Statistics:\n")
# remove whitespaces from string columns
for col in df.columns:
    if df[col].dtype == 'object':  # check if the column contains strings
        df[col] = df[col].str.strip()
    # illegal format check
    elif col in ['Age', 'Years of Education', 'Capital Gain', 'Capital Loss', 'Hours/Week']:
        df.loc[df[col] < 0, col] = 0

# replacing "?" with NaN
df.replace('?', np.nan, inplace=True)

# NaN statistics
nulls = df.isna().sum()
print("\nNaN Stats:\n\n")

for col, index in zip(nulls, df.columns):
    nan_percentage = round(col*100/len(df),2)
    print(index + "\n" + str(col) + "\n" + str(nan_percentage) + "%\n")

# fill null with mode
df['Work Class'].fillna(df['Work Class'].mode()[0], inplace=True)
df['Native Country'].fillna(df['Native Country'].mode()[0], inplace=True)

# drop null
df.dropna(subset=['Occupation'], inplace=True)

# mapping categorical values to numerals
for col in df.columns:
    if col in ['Age', 'Final Weight', 'Years of Education', 'Capital Gain', 'Capital Loss', 'Hours/Week']:
        continue
    df[col], _ = pd.factorize(df[col])

# display number of duplicated rows and null
duplicates = df.duplicated().sum()
print("\nDuplicates found: " + str(duplicates) + "\n")

df.drop_duplicates(inplace=True)

df.reset_index(drop=True, inplace=True)

print(df)

corr = df.corr()
print(corr)
df.drop(['Final Weight' , 'Type of Education' , 'Marital Status' , 'Occupation' , 'Race' , 'Native Country'], axis='columns', inplace=True)
print(df)