import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from functions import score_dataset
from sklearn.preprocessing import LabelEncoder

X = pd.read_csv("train.csv", index_col="Id")
X_test = pd.read_csv("test.csv", index_col="Id")

if __name__ == '__main__':
    # target & feature
    X.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = X["SalePrice"]
    X.drop(['SalePrice'], axis=1, inplace=True)

    # To keep things simple, we'll drop columns with missing values
    cols_with_missing = [col for col in X.columns if X[col].isnull().any()]
    X.drop(cols_with_missing, axis=1, inplace=True)
    X_test.drop(cols_with_missing, axis=1, inplace=True)

    # split full training into training and validation
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                          random_state=0)

    # only keep numerical columns in the data used for full training and submission
    X_train_numerical = X_train.select_dtypes(exclude=["object"])
    X_valid_numerical = X_valid.select_dtypes(exclude=["object"])

    print("MAE with missing values and categorical columns dropped:")
    print(score_dataset(X_train_numerical, X_valid_numerical, y_train, y_valid))

    print(X_train["Condition1"].unique())
    print(X_valid["Condition2"].unique())

    # All categorical columns
    object_cols = [col for col in X_train.columns if X_train[col].dtype == "object"]

    # Columns that can be safely label encoded
    good_label_cols = [col for col in object_cols if set(X_train[col]) == set(X_valid[col])]

    # Problematic columns that will be dropped from the dataset
    bad_label_cols = list(set(object_cols)-set(good_label_cols))

    print('Categorical columns that will be label encoded:', good_label_cols)
    print('\nCategorical columns that will be dropped from the dataset:', bad_label_cols)

    # Drop categorical columns that will not be encoded
    label_X_train = X_train.drop(bad_label_cols, axis=1)
    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

    # Apply label encoder
    label_encoder = LabelEncoder()
    for col in set(good_label_cols):
        label_X_train[col] = label_encoder.fit_transform(X_train[col])
        label_X_valid[col] = label_encoder.transform(X_valid[col])

    print("MAE from Approach 2 (Label Encoding):")
    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))

    # Get number of unique entries in each column with categorical data
    object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
    d = dict(zip(object_cols, object_nunique))

    # Print number of unique entries by column, in ascending order
    print(sorted(d.items(), key=lambda x: x[1]))
