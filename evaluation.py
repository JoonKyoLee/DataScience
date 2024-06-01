import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import numpy as np

# CSV 파일을 읽어옴
file_path = '/merged_data/final_data.csv'
data = pd.read_csv(file_path)

# 필요한 전처리 (이미 전처리가 끝난 데이터라면 생략)
# 예시: 데이터 타입 변환, 결측치 처리 등
data['사용일자'] = pd.to_datetime(data['사용일자'])

# 데이터 분할
X = data.drop(['평균이용승객수', '사용일자'], axis=1)  # 사용일자는 제거
y = data['평균이용승객수']

# 스케일링을 위한 변환
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 회귀 모델 평가를 위해 train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 회귀 모델 학습
regression_model = LinearRegression()
regression_model.fit(X_train, y_train)

# 예측
y_pred_reg = regression_model.predict(X_test)

# 회귀 모델 평가 지표 계산
mse = mean_squared_error(y_test, y_pred_reg)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)

print("Regression Evaluation:")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R²: {r2}")

# 분류 모델 준비: 이진 분류를 위해 평균이용승객수의 중간값을 기준으로 이진화
y_classification = (y > y.median()).astype(int)
y_train_clf, y_test_clf = train_test_split(y_classification, test_size=0.2, random_state=42)

# 분류 모델 학습
classification_model = LogisticRegression(max_iter=1000)
classification_model.fit(X_train, y_train_clf)
y_pred_clf = classification_model.predict(X_test)

# 분류 모델 평가 지표 계산
accuracy = accuracy_score(y_test_clf, y_pred_clf)
precision = precision_score(y_test_clf, y_pred_clf)
recall = recall_score(y_test_clf, y_pred_clf)
f1 = f1_score(y_test_clf, y_pred_clf)
conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)
class_report = classification_report(y_test_clf, y_pred_clf)

print("\nClassification Evaluation:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Classification Report:\n{class_report}")

