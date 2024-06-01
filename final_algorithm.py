import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

#CSV 파일을 읽어와 DataFrame으로 반환하는 함수
def load_data(file_path):
    return pd.read_csv(file_path)

#필요한 특징과 목표 변수를 선택하여 반환하는 함수
def select_features_and_target(df):
    features = df[['강수량', '평일', '주말']]
    target_reg = df['평균이용승객수']
    threshold = 0  
    target_clf = (target_reg > threshold).astype(int)
    return features, target_reg, target_clf

#데이터를 훈련 및 테스트 세트로 분할하는 함수
def split_data(features, target_reg, target_clf, test_size=0.2, random_state=42):
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(features, target_reg, test_size=test_size, random_state=random_state)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(features, target_clf, test_size=test_size, random_state=random_state)
    return X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_clf, X_test_clf, y_train_clf, y_test_clf

# 회귀 모델을 학습시키는 함수
def train_regression_model(X_train_reg, y_train_reg):
    reg_model = LinearRegression()
    reg_model.fit(X_train_reg, y_train_reg)
    return reg_model

#회귀 모델을 평가하는 함수
def evaluate_regression_model(reg_model, X_test_reg, y_test_reg):
    y_pred_reg = reg_model.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    return mse

# 데이터 스케일링을 위한 함수
def scale_data(X_train_clf, X_test_clf):
    scaler = StandardScaler()
    X_train_clf_scaled = scaler.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler.transform(X_test_clf)
    return X_train_clf_scaled, X_test_clf_scaled

# 분류 모델을 학습시키는 함수
def train_classification_model(X_train_clf_scaled, y_train_clf):
    clf_model = LogisticRegression(max_iter=1000)
    clf_model.fit(X_train_clf_scaled, y_train_clf)
    return clf_model

# 분류 모델을 평가하는 함수
def evaluate_classification_model(clf_model, X_test_clf_scaled, y_test_clf):
    y_pred_clf = clf_model.predict(X_test_clf_scaled)
    accuracy = accuracy_score(y_test_clf, y_pred_clf)
    return accuracy

#회귀 결과를 시각화하는 함수
def visualize_regression_results(y_test_reg, y_pred_reg):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_reg, y_pred_reg, color='blue', alpha=0.5)
    plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'k--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Regression: Actual vs Predicted')
    plt.show()

# 분류 결과의 혼동 행렬을 시각화하는 함수
def visualize_confusion_matrix(y_test_clf, y_pred_clf):
    cm = confusion_matrix(y_test_clf, y_pred_clf)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

#데이터 분포를 시각화하는 함수
def visualize_data_distribution(df):
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

# 데이터 로드
file_path = '/merged_data/final_data.csv'
df = load_data(file_path)

# 특징 및 목표 변수 선택
features, target_reg, target_clf = select_features_and_target(df)

# 데이터 분할
X_train_reg, X_test_reg, y_train_reg, y_test_reg, X_train_clf, X_test_clf, y_train_clf, y_test_clf = split_data(features, target_reg, target_clf)

# 회귀 모델 학습 및 평가
reg_model = train_regression_model(X_train_reg, y_train_reg)
mse = evaluate_regression_model(reg_model, X_test_reg, y_test_reg)
print(f'Mean Squared Error: {mse}')

# 데이터 스케일링
X_train_clf_scaled, X_test_clf_scaled = scale_data(X_train_clf, X_test_clf)

# 분류 모델 학습 및 평가
clf_model = train_classification_model(X_train_clf_scaled, y_train_clf)
accuracy = evaluate_classification_model(clf_model, X_test_clf_scaled, y_test_clf)
print(f'Accuracy: {accuracy}')

# 시각화
visualize_regression_results(y_test_reg, reg_model.predict(X_test_reg))
visualize_confusion_matrix(y_test_clf, clf_model.predict(X_test_clf_scaled))
visualize_data_distribution(df)

