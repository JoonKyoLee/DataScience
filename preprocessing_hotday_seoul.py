import pandas as pd

# 폭염일수 데이터 읽기
hot_days_seoul = pd.read_csv('origin_data/2023_서울_폭염일수.csv', encoding='utf-8')

# '지점명'을 '행정구역'으로 바꾸고 모든 값을 '경기'로 통일
hot_days_seoul['행정구역'] = '서울'

# '지점번호' 열 및 '지점명' 열 제거
hot_days_seoul = hot_days_seoul.drop(columns=['지점번호', '지점명'])

# '일자' 열을 'YYYYMMDD' 형식으로 변환
hot_days_seoul['일자'] = pd.to_datetime(hot_days_seoul['일자'], format='%Y.%m.%d').dt.strftime('%Y%m%d')

# '최고기온' 열 삭제
hot_days_seoul = hot_days_seoul.drop(columns=['최고기온'])

hot_days_seoul.to_csv('modified_data/수정된_폭염_정보(서울).csv', index=False)
