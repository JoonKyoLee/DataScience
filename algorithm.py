import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'  # macOS 사용자의 경우
# plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 사용자의 경우
plt.rcParams['axes.unicode_minus'] = False  # 그래프에서 마이너스 폰트가 깨지는 문제 해결

# 데이터 로드
df = pd.read_csv('/merged_data/final_data.csv')

# 필요한 특징과 목표 변수 선택
features = df[['강수량', '평일', '주말']]
target_reg = df['평균이용승객수']

# 분류를 위한 이진 목표 변수 생성 (평균이용승객수 임계값 설정)
threshold = 0  # 필요에 따라 임계값 조정
target_clf = (target_reg > threshold).astype(int)

# 데이터 분할 (회귀)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(features, target_reg, test_size=0.2, random_state=42)

# 데이터 분할 (분류)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(features, target_clf, test_size=0.2, random_state=42)

# 회귀 모델 학습
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# 예측 및 평가 (회귀)
y_pred_reg = reg_model.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
print(f'Mean Squared Error: {mse}')

# 데이터 스케일링 (분류)
scaler = StandardScaler()
X_train_clf_scaled = scaler.fit_transform(X_train_clf)
X_test_clf_scaled = scaler.transform(X_test_clf)

# 분류 모델 학습
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_clf_scaled, y_train_clf)

# 예측 및 평가 (분류)
y_pred_clf = clf_model.predict(X_test_clf_scaled)
accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f'Accuracy: {accuracy}')

# 회귀 모델 시각화
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Regression: Actual vs Predicted')
plt.show()

# 분류 모델 시각화 (Confusion Matrix)
cm = confusion_matrix(y_test_clf, y_pred_clf)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# 강수량과 평일/주말에 따른 승객 수 분포
plt.figure(figsize=(12, 6))

# 강수량과 평균 이용 승객수
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x='강수량', y='평균이용승객수', hue='평일', palette='viridis')
plt.title('강수량 vs 평균이용승객수')

# 평일/주말에 따른 평균 이용 승객수 분포
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='평일', y='평균이용승객수')
plt.title('평일/주말에 따른 평균이용승객수 분포')

plt.tight_layout()
plt.show()
