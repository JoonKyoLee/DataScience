import pandas as pd

# 데이터 불러오기
file_path = '/origin_data/기간내_황사.xlsx'
df = pd.read_excel(file_path)

# 1. 일자를 20240429 형식으로 변경
df['일자'] = pd.to_datetime(df['일자']).dt.strftime('%Y%m%d')

# 2. 지점번호 열 삭제
df.drop(columns=['지점번호'], inplace=True)

# 3. 서울, 인천, 또는 경기_행정구역 파일에 있는 다른 지역만 남기고 삭제
file_path2 = '/origin_data/경기_행정구역.csv'
gyeongi_info = pd.read_csv(file_path2)
gyeongi_info['시/군'] = gyeongi_info['시/군'].str.replace('시$', '', regex=True)
target_locations = ['서울', '인천'] + list(gyeongi_info['시/군'])

df = df[df['지점명'].isin(target_locations)]

# 4. 경기도에 해당하는 행정구역 값 채우기
def fill_gyeongi_district(row):
    if row['지점명'] not in ['서울', '인천']:
        match = gyeongi_info[gyeongi_info['시/군'] == row['지점명']]
        if not match.empty:
            return match.iloc[0]['행정구역']
    return row['지점명']

df['지점명'] = df.apply(fill_gyeongi_district, axis=1)


# 최종 데이터 저장
output_dir = '/modified_data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '수정_황사.csv')
data.to_csv(output_path, index=False, encoding='CP949')

print(f"파일이 저장되었습니다: {output_path}")
print(data['지점명'].unique())

# 결과 출력
print(df)
