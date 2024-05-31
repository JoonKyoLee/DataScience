import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# 한글 폰트 설정 (예: 나눔고딕)
# font_path = '../Library/Fonts/KoPubWorld Dotum Medium.ttf'
# fontprop = fm.FontProperties(fname=font_path, size=10)
# plt.rc('font', family=fontprop.get_name())
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 로드
df = pd.read_csv('merged_data/data_after_scaling.csv')

df_test = pd.read_csv('merged_data/data_after_scaling.csv')

# 사용일자를 datetime 형식으로 변환
df['사용일자'] = pd.to_datetime(df['사용일자'], format='%Y-%m-%d')

# 요일, 주말/평일, 월, 계절 변수 생성
df['요일'] = df['사용일자'].dt.dayofweek  # 0: 월요일, 6: 일요일
df['평일'] = df['요일'].apply(lambda x: 1 if x < 5 else 0)
df['주말'] = df['요일'].apply(lambda x: 1 if x >= 5 else 0)

#df['월'] = df['사용일자'].dt.month
#df['계절'] = df['월'].apply(lambda x: '봄' if 3 <= x <= 5 else ('여름' if 6 <= x <= 8 else ('가을' if 9 <= x <= 11 else '겨울')))

"""
def one_hot_encoding(origin_df, attribute):
    one_hot_encoder = OneHotEncoder(sparse_output=False)
    encoded_columns = one_hot_encoder.fit_transform(origin_df[[attribute]])

    # 인코딩된 열 이름 얻기
    encoded_column_names = one_hot_encoder.get_feature_names_out([attribute])

    # 인코딩된 열을 데이터프레임으로 변환
    encoded_df = pd.DataFrame(encoded_columns, columns=encoded_column_names)

    # 원래 데이터프레임과 인코딩된 열을 합치기
    origin_df = pd.concat([origin_df, encoded_df], axis=1)

    # 원래 열 삭제
    origin_df.drop(attribute, axis=1, inplace=True)

    return origin_df


# One-Hot Encoding
df = one_hot_encoding(df, "행정구역")

print(df.head(10))
"""
"""
# 특정 기간의 데이터 필터링
start_date = '2023-12-01'
end_date = '2024-03-31'
filtered_df = df[(df['사용일자'] >= start_date) & (df['사용일자'] <= end_date)]

# 사용일자 열 제외
filtered_df = filtered_df.drop(columns=['사용일자'])
"""

# 상관관계 행렬 계산
correlation_matrix = df.corr()

# NaN 값을 0으로 대체
correlation_matrix = correlation_matrix.fillna(0)

# 상관관계 행렬 출력
print(correlation_matrix)

df = df.drop(columns=['Unnamed: 0'])

df.to_csv('merged_data/final_data.csv')


# 상관관계 행렬 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
# plt.title('Correlation Matrix', fontproperties=fontprop)

plt.show()

"""
# 예제 데이터를 pandas DataFrame으로 로드 (가정)
df_encoded = pd.read_csv('merged_data/data_after_scaling.csv')


# 특성 변수(X) 설정
X = df_encoded.drop(columns=['평균이용승객수'])


# K-Means 클러스터링 모델 초기화 및 학습
kmeans = KMeans(n_clusters=5, random_state=42)  # 클러스터 수(K)는 데이터에 따라 조정 필요
kmeans.fit(X)

# 각 데이터 포인트의 클러스터 할당
df_encoded['Cluster'] = kmeans.labels_

# 클러스터 중심 시각화
centroids = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=300)
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 클러스터링 결과 데이터프레임 확인
print(df_encoded.head())
"""

df.drop(columns=['사용일자'], inplace=True)

# PCA 적용
pca = PCA(n_components=2)  # 주성분의 개수를 2로 설정 (예시)
principal_components = pca.fit_transform(df)

# 결과를 데이터프레임으로 변환
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# 주성분 데이터프레임에 원래의 '평균이용승객수' 컬럼 추가
pca_df['평균이용승객수'] = df['평균이용승객수'].values

# PCA 결과 시각화
plt.figure(figsize=(10, 7))
plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['평균이용승객수'], cmap='viridis')
plt.colorbar(label='평균이용승객수')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Average Passenger Usage')
plt.show()

print(pca_df)
