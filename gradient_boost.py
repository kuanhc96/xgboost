import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

X_train, X_test, y_train, y_test = train_test_split(X, y)

gradient_booster = ensemble.GradientBoostingRegressor(
    n_estimators=50,
    max_depth=8,
    learning_rate=1,
    criterion='squared_error'
)

gradient_booster.fit(X_train, y_train)
score = gradient_booster.score(X_test, y_test)
print(score)

xgb_model = xgb.XGBRegressor(
    n_estimators=50,
    reg_lambda=1,
    gamma=0,# for pruning
    max_depth=8
)

xgb_model.fit(X_train, y_train)
score = xgb_model.score(X_test, y_test)
print(score)
