import pandas as pd

# 폭염일수 데이터 읽기
hot_days_ganghwa = pd.read_csv('origin_data/2023_인천(강화)_폭염일수.csv', encoding='CP949')
hot_days_incheon = pd.read_csv('origin_data/2023_인천(인천)_폭염일수.csv', encoding='CP949')

# '지점명'을 '행정구역'으로 바꾸고 모든 값을 '인천'으로 통일
hot_days_ganghwa['행정구역'] = '인천'
hot_days_incheon['행정구역'] = '인천'

# '지점번호' 열 및 '지점명' 열 제거
hot_days_ganghwa = hot_days_ganghwa.drop(columns=['지점번호', '지점명'])
hot_days_incheon = hot_days_incheon.drop(columns=['지점번호', '지점명'])

# '일자' 열을 'YYYYMMDD' 형식으로 변환
hot_days_ganghwa['일자'] = pd.to_datetime(hot_days_ganghwa['일자'], format='%Y.%m.%d').dt.strftime('%Y%m%d')
hot_days_incheon['일자'] = pd.to_datetime(hot_days_incheon['일자'], format='%Y.%m.%d').dt.strftime('%Y%m%d')

# '최고기온' 열 삭제
hot_days_ganghwa = hot_days_ganghwa.drop(columns=['최고기온'])
hot_days_incheon = hot_days_incheon.drop(columns=['최고기온'])

# 두 데이터프레임 병합
merged_df = pd.merge(hot_days_ganghwa, hot_days_incheon, on=['일자', '행정구역'], how='outer', suffixes=('_강화', '_인천'))

# '폭염여부(O/X)' 열 결합
merged_df['폭염여부(O/X)'] = merged_df.apply(lambda row: 'O' if 'O' in [row['폭염여부(O/X)_강화'], row['폭염여부(O/X)_인천']] else 'X', axis=1)

# 불필요한 열 제거
merged_df = merged_df.drop(columns=['폭염여부(O/X)_강화', '폭염여부(O/X)_인천'])

merged_df.to_csv('modified_data/수정된_폭염_정보(인천).csv', index=False)
