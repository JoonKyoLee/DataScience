import pandas as pd

# 폭염일수 데이터 읽기
hot_days_gyeongi = pd.read_csv('../origin_data/2023_경기(수원)_폭염일수.csv', encoding='CP949')

# '지점명'을 '행정구역'으로 바꾸고 모든 값을 '경기'로 통일
hot_days_gyeongi['행정구역'] = '경기'

# '지점번호' 열 및 '지점명' 열 제거
hot_days_gyeongi = hot_days_gyeongi.drop(columns=['지점번호', '지점명'])

# '일자' 열을 'YYYYMMDD' 형식으로 변환
hot_days_gyeongi['일자'] = pd.to_datetime(hot_days_gyeongi['일자'], format='%Y.%m.%d').dt.strftime('%Y%m%d')

# '최고기온' 열 삭제
hot_days_gyeongi = hot_days_gyeongi.drop(columns=['최고기온'])

hot_days_gyeongi.to_csv('modified_data/수정된_폭염_정보(경기).csv', index=False)
