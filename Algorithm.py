from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler

import pandas as pd
import CleanData
import numpy as np


# read data
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# fill na values with average
fill_with_average = ['LotFrontage']

# fill na values with zeros
fill_with_zero = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                  'BsmtFinType2', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish',
                  'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']

data_to_clean = [train_data, test_data]

for data in data_to_clean:
    CleanData.fill_na_mean(fill_with_average, data)
    CleanData.fill_na_zero(fill_with_zero, data)
    CleanData.from_categorical_to_numerical(data)

train_data = train_data.dropna()

# IQR -> lower test score
#train_data = CleanData.subset_by_iqr(train_data, 'SalePrice', whisker_width=1.2)
train_data['SalePrice'] = np.log(train_data['SalePrice'])

test_data = test_data.fillna('0')
test_data_X = test_data.drop('Id', axis=1)

# Creating Algorithm
X = train_data.drop(['Id', 'SalePrice'], axis=1)
y = train_data['SalePrice']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# scale data
transformer = RobustScaler().fit(X_train)
X_train = transformer.transform(X_train)

transformer = RobustScaler().fit(X_test)
X_test = transformer.transform(X_test)

transformer = RobustScaler().fit(test_data_X)
test_data_X = transformer.transform(test_data_X)


# Create and fit model
param_grid = {
    'n_estimators': [1000, 1250, 1500],
    'learning_rate': [0.02, 0.05],
    'max_depth': [2, 5, 7],
    'max_features': [5, 10, 15]
}
# Create a based model
model = GradientBoostingRegressor()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           cv=3,
                           n_jobs=-1,
                           verbose=2)

# fit grid search on data
grid_search.fit(X_train, y_train)
# select best parameters and use in new model
model_best = GradientBoostingRegressor(**grid_search.best_params_)
print(grid_search.best_params_)
# fit data in new model
model_best.fit(X_train, y_train) # fit data on best parameters

# make prediction with new model
y_pred = model_best.predict(test_data_X)
y_pred = np.exp(y_pred)

# prediction dataframe
pred_dataframe = CleanData.return_submission_csv(y_pred)
