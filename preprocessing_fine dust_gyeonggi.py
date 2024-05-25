import pandas as pd

# 데이터 불러오기
file_path = '/origin_data/2304_2403_경기_미세먼지_주의보_경보_일자.csv'
df = pd.read_csv(file_path)

# 1. 번호 열 삭제
df.drop(columns=['번호'], inplace=True)

# 2. 권역 수정
df['권역'] = df['권역'].apply(lambda x: '경기' + x)

# 3. 항목 열 삭제
df.drop(columns=['항목'], inplace=True)

# 4. 발령시간과 해제시간의 중간값을 계산하여 새로운 일자 열 생성
df['발령시간'] = pd.to_datetime(df['발령시간'].str.replace('24:00', '00:00'), errors='coerce')
df['해제시간'] = pd.to_datetime(df['해제시간'].str.replace('24:00', '00:00'), errors='coerce')
df['일자'] = ((df['발령시간'] + (df['해제시간'] - df['발령시간']) / 2).dt.strftime('%Y%m%d'))

# 5. 발령시간, 발령농도, 해제시간, 해제농도 열 삭제
df.drop(columns=['발령시간', '발령농도', '해제시간', '해제농도'], inplace=True)

# 최종 데이터 저장
output_dir = '/modified_data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '경기_미세먼지_수정.csv')
data.to_csv(output_path, index=False, encoding='CP949')

print(f"파일이 저장되었습니다: {output_path}")

# 결과 출력
print(df)
