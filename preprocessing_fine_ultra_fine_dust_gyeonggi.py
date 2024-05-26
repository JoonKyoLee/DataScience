from datetime import timedelta

import pandas as pd

# 데이터 불러오기
fine_dust = pd.read_csv('origin_data/2304_2403_경기_미세먼지_주의보_경보_일자.csv')
ultra_fine_dust = pd.read_csv('origin_data/2304_2403_경기_초미세먼지_주의보_경보_일자.csv')

# 1. 번호 열 삭제
fine_dust.drop(columns=['번호'], inplace=True)
ultra_fine_dust.drop(columns=['번호'], inplace=True)

# 2. 권역 수정
fine_dust['권역'] = fine_dust['권역'].apply(lambda x: '경기' + x)
ultra_fine_dust['권역'] = ultra_fine_dust['권역'].apply(lambda x: '경기' + x)

# 3. 항목, 발령농도, 해제농도 열 삭제
fine_dust.drop(columns=['항목', '발령농도', '해제농도'], inplace=True)
ultra_fine_dust.drop(columns=['항목', '발령농도', '해제농도'], inplace=True)


def process_time_intervals(df):
    # '24' 시간을 처리하기 위한 함수 정의
    def correct_time(row):
        # 문자열 row에서 ' 24'가 포함되어 있는지 확인
        if ' 24' in row:
            # 날짜와 시간을 분리
            date, hour = row.split()
            # 시간이 '24'인 경우
            if hour == '24':
                # 날짜를 하루 증가시키고 시간을 '00:00:00'으로 설정
                corrected_time = (pd.to_datetime(date) + timedelta(days=1)).strftime('%Y-%m-%d') + ' 00:00:00'
                return corrected_time
        # 시간이 없는 경우 ':00:00'을 추가하여 유효한 시간 형식으로 변환
        # 그렇지 않으면 원래 row 반환
        return row + ':00:00' if len(row.split()) == 2 else row

    # 발령시간과 해제시간을 datetime 형식으로 변환
    df['발령시간'] = df['발령시간'].astype(str).apply(correct_time)
    df['해제시간'] = df['해제시간'].astype(str).apply(correct_time)
    df['발령시간'] = pd.to_datetime(df['발령시간'])
    df['해제시간'] = pd.to_datetime(df['해제시간'])

    # 발령시간과 해제시간 사이의 모든 날짜를 저장할 리스트 초기화
    expanded_rows = []

    # 각 행에 대해 반복
    for _, row in df.iterrows():
        # 발령시간과 해제시간 사이의 날짜 범위를 생성
        date_range = pd.date_range(start=row['발령시간'].normalize(), end=row['해제시간'].normalize(), freq='D')
        # 생성된 날짜 범위의 각 날짜에 대해 반복
        for single_date in date_range:
            # 현재 행(row)을 복사하여 새로운 행(new_row) 생성
            new_row = row.copy()
            # 새로운 행에 '일자' 열 추가하고 날짜 형식을 'YYYYMMDD'로 설정
            new_row['일자'] = single_date.strftime('%Y%m%d')
            # 새로운 행을 확장된 행 목록에 추가
            expanded_rows.append(new_row)

    # 확장된 데이터프레임 생성
    expanded_df = pd.DataFrame(expanded_rows)

    # 불필요한 열 제거
    expanded_df = expanded_df.drop(columns=['발령시간', '해제시간'])

    return expanded_df


# 함수 적용
expanded_fine_dust = process_time_intervals(fine_dust)
expanded_ultra_fine_dust = process_time_intervals(ultra_fine_dust)

expanded_fine_dust['물질명'] = '미세먼지'
expanded_ultra_fine_dust['물질명'] = '초미세먼지'

#print(expanded_fine_dust)
#print(expanded_ultra_fine_dust)

expanded_fine_dust.to_csv('modified_data/수정된_미세먼지_정보(경기).csv', index=False)
expanded_ultra_fine_dust.to_csv('modified_data/수정된_초미세먼지_정보(경기).csv', index=False)
