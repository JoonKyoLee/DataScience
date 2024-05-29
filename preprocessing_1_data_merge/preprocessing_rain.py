import pandas as pd
import os

# 원본 데이터 읽기
file_path = '../origin_data/서울경기인천_일별_시간대별_날씨정보_기온_강수량.csv'
df = pd.read_csv(file_path, encoding='CP949')

file_path2 = '../origin_data/경기_행정구역.csv'
region_df = pd.read_csv(file_path2, encoding='utf-8')

# 1. 필요 없는 열 제거
df = df.drop(columns=['기온(°C)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '증기압(hPa)', '현지기압(hPa)', '해면기압(hPa)', '일조(hr)', '일사(MJ/m2)', '적설(cm)', '3시간신적설(cm)', '전운량(10분위)', '중하층운량(10분위)', '지면온도(°C)'])

# 2. 일시 형식 통일
df['일시'] = pd.to_datetime(df['일시']).dt.strftime('%Y%m%d')

# 3. 지점명 변경 (백령도와 강화는 인천으로)
df['지점명'] = df['지점명'].replace({'백령도': '인천', '강화': '인천'})

# 4. 서울과 인천을 제외한 지역의 지점명을 경기도 행정구역 값으로 변경
region_df['시/군'] = region_df['시/군'].str.replace('시$', '', regex=True)

def fill_gyeongi_district(row):
    if row['지점명'] not in ['서울', '인천']:
        match = region_df[region_df['시/군'] == row['지점명']]
        if not match.empty:
            return match.iloc[0]['행정구역']
    return row['지점명']

df['지점명'] = df.apply(fill_gyeongi_district, axis=1)

# 5. 강수량에 따른 색상 정보 추가
def add_color(rainfall):
    if rainfall < 5:
        return 'green'
    elif rainfall < 10:
        return 'yellow'
    elif rainfall < 20:
        return 'orange'
    else:
        return 'red'

df['색'] = df['강수량(mm)'].apply(add_color)

# 최종 데이터 저장
output_dir = '/DataScience/modified_data'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, '서울경기인천_일별_시간대별_날씨정보_기온_강수량_수정.csv')
df.to_csv(output_path, index=False, encoding='CP949')

print(f"파일이 저장되었습니다: {output_path}")

# 결과 출력
print(df)
