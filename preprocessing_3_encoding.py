import pandas as pd
from tabulate import tabulate
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder

df = pd.read_csv('merged_data/data_after_remove_dirtydata.csv')

df['총승객수'] = df['승차총승객수'] + df['하차총승객수']

df['승차총승객수'] = df['총승객수']
df.rename(columns={'승차총승객수': '총이용승객수'}, inplace=True)

df.drop(columns=['하차총승객수', '총승객수'], axis=1, inplace=True)


def binary_encoding(df, attribute):
    # 'O'는 1로, 'X'는 0으로 인코딩
    df[attribute] = df[attribute].map({'O': 1, 'X': 0})
    return df


# 폭염여부, 황사관측 Encoding
df = binary_encoding(df, '황사관측')
df = binary_encoding(df, '폭염여부')


# Ordinal Encoding 함수 정의
def ordinal_encoding(df, attribute, categories):
    ordinal_encoder = OrdinalEncoder(categories=[categories])
    encoded_column = ordinal_encoder.fit_transform(df[[attribute]])

    # 열 이름 지정
    encoded_col_name = f"{attribute}_encoded"

    # 인코딩된 값을 원래 데이터 프레임에 추가
    df[encoded_col_name] = encoded_column

    # 기존에 존재하던 속성을 제거
    df.drop(columns=[attribute], axis=1, inplace=True)

    # 속성 이름 수정
    df.rename(columns={encoded_col_name: attribute}, inplace=True)

    return df


# '미세먼지', '초미세먼지', '한파특보' 열을 순서형 인코딩
df = ordinal_encoding(df, '미세먼지', ['X', '주의보', '경보'])
df = ordinal_encoding(df, '초미세먼지', ['X', '주의보', '경보'])
df = ordinal_encoding(df, '한파특보', ['X', '주의보', '경보'])

print(df)

df.to_csv("merged_data/data_after_encoding.csv", index=False)
