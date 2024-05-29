import pandas as pd

# 데이터 불러오기
file_path = '../origin_data/기간내_황사.xlsx'
df = pd.read_excel(file_path)

# 1. 일자를 20240429 형식으로 변경
df['일자'] = pd.to_datetime(df['일자']).dt.strftime('%Y%m%d')

# 2. 지점번호 열 삭제
df.drop(columns=['지점번호'], inplace=True)

# 3. 서울, 인천, 또는 수원 파일에 있는 다른 지역만 남기고 삭제
df = df[df['지점명'].isin(['서울', '인천', '수원'])]

# 새로운 열 '행정구역' 추가
df['지점명'] = df['지점명'].replace('수원', '경기')

# NaN 데이터를 X로 처리 -> 관측이 되지 않았기 때문
df['황사관측(O/X)'].fillna('X', inplace=True)

print(df)

# csv 파일로 저장
df.to_csv('modified_data/수정된_황사_정보.csv', index=False)