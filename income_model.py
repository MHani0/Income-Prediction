import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def preprocessing(df):

    # remove whitespaces from string columns
    for col in df.columns:
        col = col.strip()
        if df[col].dtype == 'object':  # check if the column contains strings
            df[col] = df[col].str.strip()
        # illegal format check
        elif col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            df[col] = df[col].astype(int)
            df.loc[df[col] < 0, col] = 0

    # replacing "?" with NaN
    df.replace('?', np.nan, inplace=True)

    # NaN statistics
    nulls = df.isna().sum()

    for col, index in zip(nulls, df.columns):
        nan_percentage = round(col * 100 / len(df), 2)

    # drop null
    df.dropna(how='any', inplace=True)

    # encoding categorical values to numerals
    for col in df.columns:
        if col in ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']:
            continue
        df[col], _ = pd.factorize(df[col])

    # display number of duplicated rows and null
    duplicates = df.duplicated().sum()

    df.drop_duplicates(inplace=True)

    df.reset_index(drop=True, inplace=True)



## Preprocessing

# read csv file
train_df = pd.read_csv("data/train_data.csv")
test_df = pd.read_csv('data/test_data.csv')

preprocessing(train_df)
preprocessing(test_df)



##Feature selection

# Lasso Regression
# Split the dataset into X (input features) and y (target variable)
X_train = train_df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
Y_train = train_df['Income']
X_test = test_df[['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']]
Y_test = test_df['Income']

# Standardize the continuous & encoded variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

# Get the remaining columns from train_df
train_remaining_columns = train_df.drop(columns=scaled_columns)
test_remaining_columns = test_df.drop(columns=scaled_columns)

# Convert X_train_scaled to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled, columns=scaled_columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=scaled_columns)

# Combine the scaled columns with the remaining columns

train_df_scaled = pd.concat([train_remaining_columns.reset_index(drop=True), X_scaled_df.reset_index(drop=True)], axis=1)
train_df_scaled.drop("Income", axis=1, inplace=True)

test_df_scaled = pd.concat([test_remaining_columns.reset_index(drop=True), X_test_scaled_df.reset_index(drop=True)], axis=1)
test_df_scaled.drop("Income", axis=1, inplace=True)

# Grid Search
param_grid = {
    'alpha': [i/100 for i in range(1, 201, 10)],
    'max_iter': [i for i in range(1000, 10001, 500)],
}

# Fit the Lasso regression model
lasso = Lasso()
grid_search = GridSearchCV(lasso, param_grid, cv=5)
grid_search.fit(train_df_scaled, Y_train)

# Get the best hyperparameters from the grid search
best_alpha = grid_search.best_params_['alpha']
best_max_iter = grid_search.best_params_['max_iter']

lasso = Lasso(alpha=best_alpha, max_iter=best_max_iter)
lasso.fit(train_df_scaled, Y_train)

# Access the coefficients
coefficients = lasso.coef_

# Select features based on coefficients that aren't zeroed by standardization
selected_features = train_df_scaled.columns[lasso.coef_ != 0]

# Filter the scaled train and test datasets based on the selected features
train_df_scaled_selected = train_df_scaled[selected_features]
test_df_scaled_selected = test_df_scaled[selected_features]



## Classification
# logistic training
logreg = LogisticRegression(max_iter=1000)
logreg.fit(train_df_scaled_selected, Y_train)

# svm training
svm = SVC()
svm.fit(train_df_scaled_selected, Y_train)

# tree training
tree = DecisionTreeClassifier()
tree.fit(train_df_scaled_selected, Y_train)

# random forest training
random_forest = RandomForestClassifier()
random_forest.fit(train_df_scaled_selected, Y_train)



## Evaluation

# logistic evaluation
logreg_pred = logreg.predict(test_df_scaled_selected)
logreg_acc = accuracy_score(Y_test, logreg_pred)
logreg_prec = precision_score(Y_test, logreg_pred)
logreg_recall = recall_score(Y_test, logreg_pred)
logreg_f1 = f1_score(Y_test, logreg_pred)
logreg_cm = confusion_matrix(Y_test, logreg_pred)

# svm evaluation
svm_pred = svm.predict(test_df_scaled_selected)
svm_acc = accuracy_score(Y_test, svm_pred)
svm_prec = precision_score(Y_test, svm_pred)
svm_recall = recall_score(Y_test, svm_pred)
svm_f1 = f1_score(Y_test, svm_pred)
svm_cm = confusion_matrix(Y_test, svm_pred)

# tree evaluation
tree_pred = tree.predict(test_df_scaled_selected)
tree_acc = accuracy_score(Y_test, tree_pred)
tree_prec = precision_score(Y_test, tree_pred)
tree_recall = recall_score(Y_test, tree_pred)
tree_f1 = f1_score(Y_test, tree_pred)
tree_cm = confusion_matrix(Y_test, tree_pred)

# random forest evaluation
rf_pred = random_forest.predict(test_df_scaled_selected)
rf_acc = accuracy_score(Y_test, rf_pred)
rf_prec = precision_score(Y_test, rf_pred)
rf_recall = recall_score(Y_test, rf_pred)
rf_f1 = f1_score(Y_test, rf_pred)
rf_cm = confusion_matrix(Y_test, rf_pred)

# printint evaluation
print('\n')
print('Logistic Regression: Accuracy = {}, Precision = {}, Recall = {}, F1-score = {}'.format(logreg_acc, logreg_prec, logreg_recall, logreg_f1))
print('SVM: Accuracy = {}, Precision = {}, Recall = {}, F1-score = {}'.format(svm_acc, svm_prec, svm_recall, svm_f1))
print('Decision Tree: Accuracy = {}, Precision = {}, Recall = {}, F1-score = {}'.format(tree_acc, tree_prec, tree_recall, tree_f1))
print('Random Forest: Accuracy = {}, Precision = {}, Recall = {}, F1-score = {}'.format(rf_acc, rf_prec, rf_recall, rf_f1))

# printing confusion matrices
print('\n')
print('Logistic Regression Confusion Matrix:\n{}'.format(logreg_cm))
print('SVM Confusion Matrix:\n{}'.format(svm_cm))
print('Decision Tree Confusion Matrix:\n{}'.format(tree_cm))
print('Random Forest Confusion Matrix:\n{}'.format(rf_cm))