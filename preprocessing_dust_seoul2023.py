import pandas as pd

# 데이터 읽기
file_path = '/origin_data/2023_서울_오존_초미세먼지_미세먼지_주의보_경보_일자.xlsx'
data = pd.read_excel(file_path)

# 1. 번호 열 제거
data = data.drop(columns=['번호'])

# 2. 물질명 열 데이터들 중에서 오존 제거, 초미세먼지, 미세먼지만 남기기
data = data[data['물질명'].isin(['초미세먼지', '미세먼지'])]

# 3. 권역 열 데이터들 '서울'로 통일
data['권역'] = '서울'

# 4. 발령시간과 해제시간의 중간 시간을 '일자' 열에 저장 (YYYYMMDD 형식)
data['발령시간'] = pd.to_datetime(data['발령시간'].str.replace('24:00', '00:00'))
data['해제시간'] = pd.to_datetime(data['해제시간'].str.replace('24:00', '00:00'))
data['일자'] = ((data['발령시간'] + (data['해제시간'] - data['발령시간']) / 2).dt.strftime('%Y%m%d'))

# 5. 발령시간과 해제시간 열 제거
data = data.drop(columns=['발령시간', '해제시간'])

# 변경된 데이터 저장
output_path = '/modified_data/2023_서울_오존_초미세먼지_미세먼지_주의보_경보_일자_modified.csv'
data.to_csv(output_path, index=False, encoding='utf-8-sig')