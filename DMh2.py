import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import time

column_names = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"]
def loadAdultData(URL):
    url = URL
    dataset = pd.read_csv(url, names=column_names, sep=",\s*", engine="python")
    dataset = dataset.replace("?", np.nan)
    dataset.dropna(inplace=True)
    # Force the data type of all columns to be strings
    for col in dataset.columns:
        dataset[col] = dataset[col].astype(str)
    return dataset

# 資料預處理
def preprocess_data(data):
    label_encoder = LabelEncoder()
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = label_encoder.fit_transform(data[col])
    return data

# 函數計算MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 計算MAPE、RMSE和R2的函數
def evaluate_performance(y_true, y_pred):
    mape = calculate_mape(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mape, rmse, r2

#Adult-----------------------------------------------------------------------------------------------------

# 載入資料
data_train = preprocess_data(loadAdultData("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"))
data_test = preprocess_data(loadAdultData("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"))
data_test = data_test.drop(1)

# 分割特徵和目標變數
X_train = data_train.drop("hours_per_week", axis=1)
y_train = data_train["hours_per_week"]
X_test = data_test.drop("hours_per_week", axis=1)
y_test = data_test["hours_per_week"]

# 標準化數值特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)      # 對訓練集進行標準化
X_test_scaled = scaler.transform(X_test)    # 使用相同的標準化轉換對測試集進行標準化

# KNN模型
knn_model = KNeighborsRegressor()
start_time = time.time()
knn_model.fit(X_train_scaled, y_train)
knn_predictions = knn_model.predict(X_test_scaled)
knn_time = time.time() - start_time
knn_mape, knn_rmse, knn_r2 = evaluate_performance(y_test, knn_predictions)

# SVR模型
svr_model = SVR()
start_time = time.time()
svr_model.fit(X_train_scaled, y_train)
svr_predictions = svr_model.predict(X_test_scaled)
svr_time = time.time() - start_time
svr_mape, svr_rmse, svr_r2 = evaluate_performance(y_test, svr_predictions)

# RandomForest模型
rf_model = RandomForestRegressor()
start_time = time.time()
rf_model.fit(X_train_scaled, y_train)
rf_predictions = rf_model.predict(X_test_scaled)
rf_time = time.time() - start_time
rf_mape, rf_rmse, rf_r2 = evaluate_performance(y_test, rf_predictions)

# XGBoost模型
xgb_model = XGBRegressor()
start_time = time.time()
xgb_model.fit(X_train_scaled, y_train)
xgb_predictions = xgb_model.predict(X_test_scaled)
xgb_time = time.time() - start_time
xgb_mape, xgb_rmse, xgb_r2 = evaluate_performance(y_test, xgb_predictions)

# 輸出結果
print("KNN Metrics:")
print(f"MAPE: {knn_mape}, RMSE: {knn_rmse}, R2: {knn_r2}")
print(f"Time: {knn_time} seconds")

print("\nSVR Metrics:")
print(f"MAPE: {svr_mape}, RMSE: {svr_rmse}, R2: {svr_r2}")
print(f"Time: {svr_time} seconds")

print("\nRandomForest Metrics:")
print(f"MAPE: {rf_mape}, RMSE: {rf_rmse}, R2: {rf_r2}")
print(f"Time: {rf_time} seconds")

print("\nXGBoost Metrics:")
print(f"MAPE: {xgb_mape}, RMSE: {xgb_rmse}, R2: {xgb_r2}")
print(f"Time: {xgb_time} seconds")

#Adult-----------------------------------------------------------------------------------------------------

#Boston-----------------------------------------------------------------------------------------------------

# 載入Boston Housing資料集的替代方法
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 將資料轉換成DataFrame
columns = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]
data = pd.DataFrame(data, columns=columns)
data['MEDV'] = target.astype(float) # 將目標變數轉換為浮點數型態，這是因為模型的輸出通常期望是浮點數

# 分割特徵和目標變數
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# XGBoost模型
xgb_model = XGBRegressor()

# K-fold cross validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

fold_results = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 訓練模型
    xgb_model.fit(X_train, y_train)

    # 預測
    predictions = xgb_model.predict(X_test)

    # 評估績效
    mape, rmse, r2 = evaluate_performance(y_test, predictions)

    # 將每個fold的績效指標記錄下來
    fold_results.append({'MAPE': mape, 'RMSE': rmse, 'R2': r2})

# 計算平均績效
avg_results = pd.DataFrame(fold_results).mean()

# 輸出每一個fold的預測績效以及5 folds的平均績效
print("Performance for each fold:")
print(pd.DataFrame(fold_results))
print("\nAverage Performance:")
print(avg_results)

#Boston-----------------------------------------------------------------------------------------------------