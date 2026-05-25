# Gözetimli Öğrenme Ödevi
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Automobile_data.csv", na_values="?")

print(df.info())
print(df.head())

nan_values = df.isnull().sum()
nan_values = nan_values[nan_values > 0].sort_values(ascending=False)
print(nan_values.keys())

nan_values_cols = ['normalized-losses', 'bore', 'stroke', 'price', 'num-of-doors', 'horsepower', 'peak-rpm']

for i in nan_values_cols:
    print(df[df[i].isna()])

print(df.describe())

df = df.dropna(subset=["price"])

df["num-of-doors"] = df["num-of-doors"].fillna(df["num-of-doors"].mode()[0])

num_nan_cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm"]

for col in num_nan_cols:

    df[col] = pd.to_numeric(df[col], errors="coerce")
    col_medyani = df[col].median()
    df[col] = df[col].fillna(col_medyani)

nan_values = df.isnull().sum()
nan_values = nan_values[nan_values > 0].sort_values(ascending=False)
print(nan_values)

print(df.info())

cat_cols = df.select_dtypes(include="object")

print(cat_cols)

for i in cat_cols:
    print(df[i].value_counts())
    print(df[i].unique())


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# import math

# def plot_all_histograms(df):
#     num_cols = df.select_dtypes(include=[np.number]).columns
#     n_cols = 3
#     n_rows = math.ceil(len(num_cols) / n_cols)

#     plt.figure(figsize = (5 * n_cols, 4 * n_rows))

#     for i,col in enumerate(num_cols,1):
#         plt.subplot(n_rows,n_cols,i)
#         sns.histplot(df[col],bins=30,kde = True)
#         plt.title(f"{col}")
#         plt.xlabel("")
#         plt.ylabel("")

#     plt.tight_layout()
#     plt.show()


# print(plot_all_histograms(df))

X = df.drop("price", axis = 1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)

print(X_train)

ordinal_cols = ["num-of-doors", "num-of-cylinders"]
binary_cols = ["fuel-type", "aspiration", "engine-location"]
nominal_cols = ["make", "body-style", "drive-wheels", "engine-type", "fuel-system"]

door_cats = ["two", "four"]
cylinder_cats = ["two", "three", "four", "five", "six", "eight", "twelve"]

preprocessor = ColumnTransformer(transformers=[
    ("ordinal", OrdinalEncoder(categories=[door_cats, cylinder_cats]), ordinal_cols),
    ("binary", OneHotEncoder(drop="first", handle_unknown="ignore"), binary_cols),
    ("nominal", OneHotEncoder(handle_unknown="ignore"), nominal_cols)
    ],
    remainder="passthrough"
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

X_cols = preprocessor.get_feature_names_out()

X_train = pd.DataFrame(X_train, columns=X_cols)
X_test = pd.DataFrame(X_test, columns=X_cols)

print(X_train.info())

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_cols = preprocessor.get_feature_names_out()

X_train = pd.DataFrame(X_train_scaled, columns=X_cols)
X_test = pd.DataFrame(X_test_scaled, columns=X_cols)

#sns.pairplot(df)
#plt.show()

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def evaluate_model(true,pred):
    mae = mean_absolute_error(true,pred)
    mse = mean_squared_error(true,pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(true,pred)

    return mae,mse,rmse,r2

models = {
    "Linear" : LinearRegression(),
    "Ridge" : RidgeCV(),
    "Lasso" : LassoCV(),
    "KNeighbours" : KNeighborsRegressor(),
    "Decision Tree" : DecisionTreeRegressor(random_state=23),
    "Random Forest" : RandomForestRegressor(random_state=23),
    "Ada Boost" : AdaBoostRegressor(random_state=23),
    "Gradient Boost" : GradientBoostingRegressor(random_state=23),
    "XGBoost" :  XGBRegressor(random_state=23, verbose = -1),
    "LightGBM" : LGBMRegressor(random_state=23, verbose = -1)
}

results = []

for name, model in models.items():
    
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    train_mae, train_mse, train_rmse, train_r2 = evaluate_model(y_train, y_train_pred)

    y_test_pred = model.predict(X_test)
    test_mae, test_mse, test_rmse, test_r2 = evaluate_model(y_test, y_test_pred)

    print(f"--- {name} Modeli Sonuçları ---")
    print("EĞİTİM (TRAIN) PERFORMANSI:")
    print(f"MAE: {train_mae:.4f} | MSE: {train_mse:.4f} | RMSE: {train_rmse:.4f} | R2 Score: {train_r2:.4f}")
    
    print("TEST PERFORMANSI:")
    print(f"MAE: {test_mae:.4f} | MSE: {test_mse:.4f} | RMSE: {test_rmse:.4f} | R2 Score: {test_r2:.4f}")
    
    r2_diff = train_r2 - test_r2
    print(f"Train - Test R2 Farkı (Overfitting Göstergesi): {r2_diff:.4f}")
    print("_" * 50, "\n")

    results.append([name, train_r2, test_r2, r2_diff])

results_dftrain = pd.DataFrame(results, columns=["Model", "Train R2 Score", "Test R2 Score", "R2 Difference"]).sort_values(by="Test R2 Score", ascending=False)

print("TÜM MODELLERİN KARŞILAŞTIRMALI TABLOSU:")
print(results_dftrain)

from sklearn.model_selection import GridSearchCV

# Random Forest için grid search
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, None],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}
rf = RandomForestRegressor(random_state=23)
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='r2', n_jobs=-1)
grid_rf.fit(X_train, y_train)
y_test_pred = grid_rf.predict(X_test)
y_train_pred = grid_rf.predict(X_train)

print("Best RF params:", grid_rf.best_params_)

print("Test R2:", r2_score(y_test, y_test_pred))
print("Train R2:", r2_score(y_train, y_train_pred))

xgboost_params = {
    "learning_rate" : [0.01,0.1],
    "max_depth" : [5,8,12,20,30],
    "n_estimators" : [100,200,300,500],
    "colsample_bytree" : [0.3,0.4,0.5,0.7,0.9,1],
}

grid_xgb = GridSearchCV(estimator=XGBRegressor(random_state = 23), param_grid=xgboost_params, cv=5, n_jobs=-1)
grid_xgb.fit(X_train, y_train)

y_test_pred = grid_xgb.predict(X_test)
y_train_pred = grid_xgb.predict(X_train)

print("Best XGBoost params:", grid_xgb.best_params_)

print("Test R2:", r2_score(y_test, y_test_pred))
print("Train R2:", r2_score(y_train, y_train_pred))