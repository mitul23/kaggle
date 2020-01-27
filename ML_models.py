####-----------------------------------------------------------##########################
# Approach 1: drop columns with missing values and train the model (just numerical values)
missing_col_train = (X_train.isnull().sum())
print(missing_col_train[missing_col_train > 0])

# get names of columns with missing values
missing_col_names = [col for col in X_train.columns if X_train[col].isnull().any()]
print(missing_col_names)

# drop columns in training and validation data
reduced_X_train = X_train.drop(missing_col_names, axis=1)
reduced_X_valid = X_valid.drop(missing_col_names, axis=1)

# model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
# model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
# model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
# model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
# model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

# models = [model_1, model_2, model_3, model_4, model_5]

# for i in range(len(models)):
#    mae = score_model(models[i])
#    print("Model %d MAE: %d" % (i+1, mae))

print("MAE (Drop columns with missing values):")
print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

####-----------------------------------------------------------##########################
# Approach 2: Imputate missing values with mean of each cols
imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("\n")
print("MAE (Imputation with mean):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

imputer = SimpleImputer(strategy="median")
imputed_X_train = pd.DataFrame(imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(imputer.transform(X_valid))
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
print("\n")
print("MAE (Imputation with median):")
print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

# transform test set and make prediction
imputer_X_test = pd.DataFrame(imputer.transform(test))
model = RandomForestRegressor(n_estimators=100, random_state=0)
model.fit(imputed_X_train, y_train)
pred_valid = model.predict(imputed_X_valid)
pred_test = model.predict(imputer_X_test)
output = pd.DataFrame({"Id": test.index,
                       "SalePrice": pred_test})
print("\n")
print("training data and prediction: ")
print(pred_valid.shape)
print(pd.DataFrame({"Id": y_train.index,
                    "Actual SalePrice": y_train[1]}))
print("\n")
print("Predictions on test set: ")
print(output.head())
