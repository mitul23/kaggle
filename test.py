

model = RandomForestRegressor(n_estimators=100, random_state=0)


pred_test = model.predict(imputer_X_test)
output = pd.DataFrame({"Id": test.index,
                       "SalePrice": pred_test})
print("\n")
print("Predictions on test set: ")
print(output.head())
