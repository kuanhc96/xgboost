import pandas as pd
import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = pd.read_csv("/Users/kuanhc96/Desktop/pyimagesearch/DL108-XGBoost/realtor-data.zip.csv")
data = data.dropna() # pd.DataFrame

y = data["price"].values # price is the target column
X = data.drop(["price"], axis=1) # drop the price column, since that is the target column
X = X.select_dtypes(exclude=['object']) # only keep the numerical columns. The non-numerical columns (object) will need one-hot encoding

X_train, X_test, y_train, y_test = train_test_split(X, y)

xgb_model = xgb.XGBRegressor(
    n_estimators=100, # number of trees
    reg_lambda=1, # lambda value for L2 regularization
    gamma=0, # gamma min reduction of 
    max_depth=4
)

xgb_model.fit(X_train, y_train, verbose=True)

feature_importance = pd.DataFrame(xgb_model.feature_importances_.reshape(1, -1), columns=X.columns)
print(feature_importance)

predictions = xgb_model.predict(X_test)
mean_squared_error = mean_squared_error(predictions, y_test)
print(mean_squared_error)

score = xgb_model.score(X_test, y_test)
print(score)
