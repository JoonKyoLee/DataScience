import pandas as pd
import numpy as np

cold = pd.read_excel('../origin_data/한파일수.xlsx')
cold_region = pd.read_csv('../origin_data/한파_지점_행정구역.csv')

# '일시' 열을 문자열로 변환
cold['일시'] = cold['일시'].astype(str)


# 날짜 형식을 변환하는 함수 정의
def convert_date_format(date_str):
    parts = date_str.split('-')
    year = parts[0]
    month = parts[1].zfill(2)  # 월을 두 자리로 맞춤
    day = parts[2].zfill(2)    # 일을 두 자리로 맞춤
    return f"{year}{month}{day}"


# 날짜 형식을 변환 (2023.4.1 -> 20230401)
cold['일시'] = cold['일시'].apply(convert_date_format)

# 각 지역 뒤의 괄호를 제거
cold['지점'] = cold['지점'].str.replace(r'\(.*\)', '', regex=True)

# 서울, 경기, 인천 외의 지역은 삭제
cold = cold[cold['지점'].isin(cold_region['지점'])]

# 서울의 경우에는 서울, 인천의 경우에는 인천, 경기의 경우에는 지역에 맞는 행정구역 삽입
cold = pd.merge(cold, cold_region, on='지점', how='left')

cold = cold[['일시', '지점', '한파특보(O/X)', '행정구역']]


def majority_vote(group):
    return group.value_counts().idxmax()


result = cold.groupby(['일시', '행정구역'])['한파특보(O/X)'].apply(majority_vote).reset_index()

result.to_csv('modified_data/수정된_한파_정보.csv', index=False)
