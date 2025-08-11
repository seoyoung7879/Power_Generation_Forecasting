import pandas as pd
import numpy as np

# ===================================================================
# 파생변수 계산 함수 (기존 컬럼을 직접 사용하는 간결한 버전)
# ===================================================================

def calc_air_density_simplified(row):
    """
    기존의 '현지기압(hPa)'과 '증기압(hPa)' 컬럼을 직접 사용해 대기 밀도를 계산합니다.
    """
    # 필요한 값: 기온(°C), 현지기압(hPa), 증기압(hPa)
    T_celsius = row['기온(°C)']
    P_hpa = row['현지기압(hPa)']  # <-- 현지기압 직접 사용
    e_hpa = row['증기압(hPa)']    # <-- 증기압 직접 사용

    # 1. 건조공기압력 (hPa) 계산
    pd_hpa = P_hpa - e_hpa

    # 2. 습윤공기 밀도(rho, kg/m³) 계산
    Rd = 287.058  # 건조공기 기체상수 J/(kg·K)
    Rv = 461.495  # 수증기 기체상수 J/(kg·K)
    T_kelvin = T_celsius + 273.15 # 섭씨를 켈빈으로 변환

    # 이상기체 상태방정식을 이용한 밀도 계산 (압력은 Pa 단위로 변환)
    rho = (pd_hpa * 100) / (Rd * T_kelvin) + (e_hpa * 100) / (Rv * T_kelvin)
    
    return rho

def calc_absolute_humidity_simplified(row):
    """
    기존의 '기온(°C)'과 '증기압(hPa)' 컬럼을 직접 사용해 절대 습도를 계산합니다.
    """
    # 필요한 값: 기온(°C), 증기압(hPa)
    T_celsius = row['기온(°C)']
    e_hpa = row['증기압(hPa)']  # <-- 증기압 직접 사용
    T_kelvin = T_celsius + 273.15

    # 절대습도(AH, g/m³) 계산
    # 공식: AH = (e_pa * 1000) / (Rv * T_kelvin) = 216.7 * e_hpa / T_kelvin
    # 216.7 은 (100 * 1000) / 461.5 를 근사한 값입니다.
    abs_humidity = 216.7 * (e_hpa / T_kelvin)
    
    return abs_humidity

# --- 메인 코드 실행 ---

# 1. 데이터 불러오기
# 파일 경로가 다른 경우, 이 부분을 수정해주세요.
file_path = '풍력_공모전/풍력_발전량_예측_모델/1. 데이터분석/전처리_과정_데이터/전처리완료_기상과풍력.csv'
try:
    df = pd.read_csv(file_path)
    print(f"'{file_path}' 파일 로딩 성공! (데이터: {len(df):,}건)")
except FileNotFoundError:
    print(f"[오류] '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인해주세요.")
    # 파일이 없으면 코드 실행 중단
    exit()

# 2. 파생변수 추가
print("\n파생변수 계산을 시작합니다...")

# 대기 밀도 컬럼 추가
df['air_density'] = df.apply(calc_air_density_simplified, axis=1)
print("  - 'air_density' 컬럼 생성 완료")

# 절대 습도 컬럼 추가
df['absolute_humidity'] = df.apply(calc_absolute_humidity_simplified, axis=1)
print("  - 'absolute_humidity' 컬럼 생성 완료")


# 3. 결과 확인
print("\n[생성된 파생변수 샘플 확인]")
print(df[['기온(°C)', '현지기압(hPa)', '증기압(hPa)', 'air_density', 'absolute_humidity']].head())


# 4. 최종 데이터 저장
output_path = '풍력_공모전/풍력_발전량_예측_모델/1. 데이터분석/최종사용_데이터/파생변수추가_기상과풍력.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n✅ 모든 작업 완료! '{output_path}' 파일로 저장되었습니다.")

