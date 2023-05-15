import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


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

    # drop null
    df.dropna(how='any', inplace=True)

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


# Lasso Regression

# Split the dataset into X (input features) and y (target variable)
X = train_df.drop('Income', axis=1)
y = train_df['Income']

# Standardize the continuous & encoded variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the Lasso regression model
alpha = 0.01  # Hyperparameter
lasso = Lasso(alpha=alpha)  # Set the regularization strength (alpha)
lasso.fit(X_scaled, y)

max_iter = 1000  # Hyperparameter
lasso = Lasso(alpha=alpha, max_iter=max_iter)
lasso.fit(X_scaled, y)

# Access the coefficients
coefficients = lasso.coef_

# Select features based on coefficients that aren't zeroed by standardization
selected_features = X.columns[lasso.coef_ != 0]
print("Selected features:", selected_features)

for cols in train_df.columns:
    if cols in selected_features or cols == 'Income':
        continue
    train_df.drop(cols, axis=1, inplace=True)

print(train_df)
