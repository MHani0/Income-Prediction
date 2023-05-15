import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def preprocessing(df):

    print("\nPre-processing Statistics:\n")
    # remove whitespaces from string columns
    for col in df.columns:
        col = col.strip()
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
X_train = train_df.drop('Income', axis=1)
Y_train = train_df['Income']
X_test = test_df.drop('Income', axis=1)
Y_test = test_df['Income']

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# Standardize the continuous & encoded variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
print(X_scaled)

# Fit the Lasso regression model
alpha = 0.01  # Hyperparameter
lasso = Lasso(alpha=alpha)  # Set the regularization strength (alpha)
lasso.fit(X_scaled, Y_train)

max_iter = 1000  # Hyperparameter
lasso = Lasso(alpha=alpha, max_iter=max_iter)
lasso.fit(X_scaled, Y_train)

# Access the coefficients
coefficients = lasso.coef_

# Select features based on coefficients that aren't zeroed by standardization
selected_features = X_train.columns[lasso.coef_ != 0]
print("Selected features:", selected_features)

for cols in train_df.columns:
    if cols in selected_features or cols == 'Income':
        continue
    train_df.drop(cols, axis=1, inplace=True)


print(train_df)


logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)

svm = SVC()
svm.fit(X_train, Y_train)

tree = DecisionTreeClassifier()
tree.fit(X_train, Y_train)

logreg_pred = logreg.predict(X_test)
logreg_acc = accuracy_score(Y_test, logreg_pred)
logreg_prec = precision_score(Y_test, logreg_pred)
logreg_recall = recall_score(Y_test, logreg_pred)
logreg_f1 = f1_score(Y_test, logreg_pred)
logreg_cm = confusion_matrix(Y_test, logreg_pred)

svm_pred = svm.predict(X_test)
svm_acc = accuracy_score(Y_test, svm_pred)
svm_prec = precision_score(Y_test, svm_pred)
svm_recall = recall_score(Y_test, svm_pred)
svm_f1 = f1_score(Y_test, svm_pred)
svm_cm = confusion_matrix(Y_test, svm_pred)

tree_pred = tree.predict(X_test)
tree_acc = accuracy_score(Y_test, tree_pred)
tree_prec = precision_score(Y_test, tree_pred)
tree_recall = recall_score(Y_test, tree_pred)
tree_f1 = f1_score(Y_test, tree_pred)
tree_cm = confusion_matrix(Y_test, tree_pred)

print('Logistic Regression: Accuracy = {}, Precision = {}, Recall = {}, F1-score = {}'.format(logreg_acc, logreg_prec, logreg_recall, logreg_f1))
print('SVM: Accuracy = {}, Precision = {}, Recall = {}, F1-score = {}'.format(svm_acc, svm_prec, svm_recall, svm_f1))
print('Decision Tree: Accuracy = {}, Precision = {}, Recall = {}, F1-score = {}'.format(tree_acc, tree_prec, tree_recall, tree_f1))

print('Logistic Regression Confusion Matrix:\n{}'.format(logreg_cm))
print('SVM Confusion Matrix:\n{}'.format(svm_cm))
print('Decision Tree Confusion Matrix:\n{}'.format(tree_cm))