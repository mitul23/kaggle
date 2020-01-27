from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    train = pd.read_csv("train.csv")

    # plt.scatter(train["SalePrice"], train["YearBuilt"])
    # plt.show()
kf = KFold(n_splits=5, shuffle=True, random_state=123)


def get_kfold_rmse(train):
    mse_scores = []
    for train_index, test_index in kf.split(train):
        train = train.fillna(0)
        feats = [x for x in train.columns if x not in [
            'Id', 'SalePrice', 'RoofStyle', 'CentralAir']]
        fold_train, fold_test = train.loc[train_index], train.loc[test_index]
        # Fit the data and make predictions
        # Create a Random Forest object
        rf = RandomForestRegressor(n_estimators=10, min_samples_split=10, random_state=123)
        # Train a model
        rf.fit(X=fold_train[feats], y=fold_train['SalePrice'])
        # Get predictions for the test set
        pred = rf.predict(fold_test[feats])
        fold_score = mean_squared_error(fold_test['SalePrice'], pred)
        mse_scores.append(np.sqrt(fold_score))
    return round(np.mean(mse_scores) + np.std(mse_scores), 2)


# Look at the initial RMSE
print('RMSE before feature engineering:', get_kfold_rmse(train))
# Find the total area of the house - improves RMSE
train['TotalArea'] = train['TotalBsmtSF'] + train['FirstFlrSF'] + train['SecondFlrSF']
print('RMSE with total area:', get_kfold_rmse(train))
# Find the area of the garden - improves RMSE
train['GardenArea'] = train['LotArea'] - train['FirstFlrSF']
print('RMSE with garden area:', get_kfold_rmse(train))
# Find total number of bathrooms - do not improve RMSE - not a good feature
train['TotalBath'] = train["FullBath"] + train["HalfBath"]
print('RMSE with number of bathrooms:', get_kfold_rmse(train))

# dealing with binary features (categorical features with only two categories) it is suggested to apply label encoder only
# Concatenate train and test together
houses = pd.concat([train, test])

# Label encoder
le = LabelEncoder()
# Create new features
ohe = pd.get_dummies(houses['RoofStyle'], prefix='RoofStyle')
houses['CentralAir_enc'] = le.fit_transform(houses.CentralAir)  # is binary so apply LabelEncoder
# Concatenate OHE features to houses
houses = pd.concat([houses, ohe], axis=1)


def test_mean_target_encoding(train, test, target, categorical, alpha=5):
    # Calculate global mean on the train data
    global_mean = train[target].mean()

    # Group by the categorical feature and calculate its properties
    train_groups = train.groupby(categorical)
    category_sum = train_groups[target].sum()
    category_size = train_groups.size()

    # Calculate smoothed mean target statistics
    train_statistics = (category_sum + global_mean * alpha) / (category_size + alpha)

    # Apply statistics to the test data and fill new categories
    test_feature = test[categorical].map(train_statistics).fillna(global_mean)
    return test_feature.values


def train_mean_target_encoding(train, target, categorical, alpha=5):
    # Create 5-fold cross-validation
    kf = KFold(n_splits=5, random_state=123, shuffle=True)
    train_feature = pd.Series(index=train.index)

    # For each folds split
    for train_index, test_index in kf.split(train):
        cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]

        # Calculate out-of-fold statistics and apply to cv_test
        cv_test_feature = test_mean_target_encoding(cv_train, cv_test, target, categorical, alpha)

        # Save new feature for this particular fold
        train_feature.iloc[test_index] = cv_test_feature
    return train_feature.values


def mean_target_encoding(train, test, target, categorical, alpha=5):

    # Get the train feature
    train_feature = train_mean_target_encoding(train, target, categorical, alpha)

    # Get the test feature
    test_feature = test_mean_target_encoding(train, test, target, categorical, alpha)

    # Return new features to add to the model
    return train_feature, test_feature


# Create mean target encoded feature
train['RoofStyle_enc'], test['RoofStyle_enc'] = mean_target_encoding(train=train,
                                                                     test=test,
                                                                     target='SalePrice',
                                                                     categorical='RoofStyle',
                                                                     alpha=10)
# Look at the encoding
print(test[['RoofStyle', 'RoofStyle_enc']].drop_duplicates())
