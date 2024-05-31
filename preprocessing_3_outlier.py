import pandas as pd
import matplotlib.pyplot as plt


def removeOutliers(df_origin, df_simplified, column):
    
    # Calculate IQR
    q1 = df_simplified[column].quantile(0.25)
    q3 = df_simplified[column].quantile(0.75)
    iqr = q3 - q1

    # IQR range
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    # find outlier dates
    df_outlier = df_simplified[(df_simplified[column] < lower_bound) | (df_simplified[column] > upper_bound)]
    outlier_dates = df_outlier['사용일자'].unique()
    print("\noutliers:")
    print(df_outlier)
    
    # remove outliers from original dataframe
    result_df = df_origin
    for i, row in result_df.iterrows():
        
        if row['사용일자'] in outlier_dates:
            result_df.drop(index=i, inplace=True)

    return result_df



def visualizeOutliers(data_before, data_after, column):
    
    plt.figure(figsize=(12, 6))
    plt.rcParams['font.family'] = 'Malgun Gothic'
    
    # Visualize data before removal
    plt.subplot(1, 2, 1)
    plt.title(f"이상치 제거 전 ({column})")
    plt.boxplot(data_before[column])
    plt.ylabel("값")
    
    # Visualize data after removal
    plt.subplot(1, 2, 2)
    plt.title(f"이상치 제거 후 ({column})")
    plt.boxplot(data_after[column])
    plt.ylabel("값")
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    
    df = pd.read_csv('merged_data/data_after_remove_dirtydata.csv')
    
    ## feature engineering before preprocessing outlier
    # 승차 총 승객수와 하차 총 승객수를 합한 총 승객수 열 생성
    df['총승객수'] = df['승차총승객수'] + df['하차총승객수']

    # 승차 승객수 자리에 총 승객수를 넣고 승차/하차 승객수를 drop
    df['승차총승객수'] = df['총승객수']
    df.rename(columns={'승차총승객수': '총이용승객수'}, inplace=True)
    df.drop(columns=['하차총승객수', '총승객수'], axis=1, inplace=True)

    # 사용일자를 datetime 형식으로 변환
    df['사용일자'] = pd.to_datetime(df['사용일자'], format='%Y%m%d')
    
    
    
    # Simplify dataframe to find outlier
    df_simplified = df[df['사용일자'].dt.dayofweek < 5]    # weekdays
    df_simplified = df_simplified[['사용일자', '총이용승객수']]

    # '사용일자'별로 '총승객수'를 그룹화하여 합계 계산
    df_simplified = df_simplified.groupby('사용일자')['총이용승객수'].sum().reset_index()

    # 이상치 제거할 열 선택
    column = '총이용승객수'

    # 이상치 제거 및 결과 저장
    result_df = removeOutliers(df, df_simplified, column)
    result_df.to_csv('merged_data/data_after_remove_outlier.csv', index=False)



    # 시각화를 위한 result_df 처리
    result_df_simplified = result_df[result_df['사용일자'].dt.dayofweek < 5]    # weekdays
    result_df_simplified = result_df_simplified.groupby('사용일자')['총이용승객수'].sum().reset_index()

    # 이상치 제거된 데이터 비교 시각화
    visualizeOutliers(df_simplified, result_df_simplified, column)
    
    # 지수 표기법 대신 일반 표기법으로 출력하도록 설정 변경
    pd.set_option('display.float_format', '{:.2f}'.format)

    # 이상치 제거 전후의 통계적 요약 출력
    print("\n'총이용승객수' 이상치 제거 전:")
    print(df[column].describe())
    print("\n'총이용승객수' 이상치 제거 후:")
    print(result_df[column].describe())
