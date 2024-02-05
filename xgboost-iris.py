import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y)

xgb_model = xgb.XGBRegressor(
    n_estimators=100, # number of trees
    reg_lambda=1, # lambda value for L2 regularization
    gamma=0, # gamma min reduction of 
    max_depth=4
)

# fit the training data in the model
xgb_model.fit(X_train, y_train)

feature_importances = pd.DataFrame(xgb_model.feature_importances_.reshape(1, -1), columns = iris.feature_names)

predictions = xgb_model.predict(X_test)
error = mean_squared_error(predictions, y_test)
print(error)