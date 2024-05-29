import pandas as pd

# 데이터 읽기
data = pd.read_excel('../origin_data/서울_오존_초미세먼지_미세먼지_주의보_경보_일자.xlsx')

# 2. 물질명 열 데이터들 중에서 오존 제거, 초미세먼지, 미세먼지만 남기기
data = data[data['물질명'].isin(['초미세먼지', '미세먼지'])]

# 3. 권역 열 데이터들 '서울'로 통일
data['권역'] = '서울'

# 4. 발령시간과 해제시간 변경(YYYYMMDD 형식)
data['발령시간'] = pd.to_datetime(data['발령시간'].str.replace('24:00', '00:00'))
data['해제시간'] = pd.to_datetime(data['해제시간'].str.replace('24:00', '00:00'))

# 발령시간과 해제시간 사이의 모든 날짜를 저장할 리스트 초기화
expanded_rows = []

# 각 행에 대해 반복
for _, row in data.iterrows():  # 각 행에 대해 반복
    # 발령시간과 해제시간 사이의 날짜 범위를 생성 (freq:'D' -> Day 단위)
    date_range = pd.date_range(start=row['발령시간'].normalize(), end=row['해제시간'].normalize(), freq='D')

    for single_date in date_range:  # 생성된 날짜 범위의 각 날짜에 대해 반복
        new_row = row.copy()  # 현재 행을 복사
        new_row['일자'] = single_date.strftime('%Y%m%d')  # 날짜 형식을 'YYYYMMDD'로 설정
        # 새로운 행을 확장된 행 목록에 추가
        expanded_rows.append(new_row)

# 확장된 데이터프레임 생성
data_expanded = pd.DataFrame(expanded_rows)

# 불필요한 열 제거
data_expanded = data_expanded.drop(columns=['발령시간', '해제시간'])

fine_dust_seoul = data_expanded[data_expanded['물질명'] == '미세먼지']
ultra_fine_dust_seoul = data_expanded[data_expanded['물질명'] == '초미세먼지']

# csv 파일로 저장
fine_dust_seoul.to_csv('modified_data/수정된_미세먼지_정보(서울).csv', index=False)
ultra_fine_dust_seoul.to_csv('modified_data/수정된_초미세먼지_정보(서울).csv', index=False)
