import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

#CSV 파일을 읽어와 DataFrame으로 반환하는 함수
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['사용일자'] = pd.to_datetime(data['사용일자'])
    return data

# 데이터 전처리 함수 (필요한 경우에만 적용)
def preprocess_data(data):
    X = data.drop(['평균이용승객수', '사용일자'], axis=1)  
    y = data['평균이용승객수']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

#데이터를 훈련 및 테스트 세트로 분할하는 함수
def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# 회귀 모델을 학습시키는 함수
def train_regression_model(X_train, y_train):
    regression_model = LinearRegression()
    regression_model.fit(X_train, y_train)
    return regression_model

#회귀 모델을 평가하는 함수
def evaluate_regression_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, rmse, mae, r2

#분류 모델을 학습시키는 함수
def train_classification_model(X_train, y_train):
    classification_model = LogisticRegression(max_iter=1000)
    classification_model.fit(X_train, y_train)
    return classification_model

#분류 모델을 평가하는 함수
def evaluate_classification_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    return accuracy, precision, recall, f1, conf_matrix, class_report

# 데이터 로드
file_path = '/merged_data/final_data.csv'
data = load_data(file_path)

# 데이터 전처리
X, y = preprocess_data(data)

# 데이터 분할
X_train, X_test, y_train, y_test = split_data(X, y)

# 회귀 모델 학습
regression_model = train_regression_model(X_train, y_train)

# 회귀 모델 평가
mse, rmse, mae, r2 = evaluate_regression_model(regression_model, X_test, y_test)
print("Regression Evaluation:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# 분류 모델 학습
y_classification = (y > y.median()).astype(int)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(X, y_classification)
classification_model = train_classification_model(X_train_clf, y_train_clf)

# 분류 모델 평가
accuracy, precision, recall, f1, conf_matrix, class_report = evaluate_classification_model(classification_model, X_test_clf, y_test_clf)
print("\nClassification Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")
