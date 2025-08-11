# -*- coding: utf-8 -*-
import math
import os
from datetime import datetime
import pandas as pd
import requests
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==============================================================================
# 설정 (Configurations)
# ==============================================================================
import os

# ✅ 현재 파일 위치 기준 BASE_DIR 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CONFIG = {
    "api": {
        "service_key": "q76Ri/TQYQG6deBQQyxGFLVWEefNtKOGMFG7a7UhOMzc9ohDtZbVlX4WDGkJSgASOHebDmhJ3Lk9axDc5Pv00w==",
        "base_url": "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    },
    "files": {
        # BASE_DIR/../data/격자예보_관측지점매핑.csv
        "grid_mapping_csv": os.path.join(BASE_DIR, "..", "data", "격자예보_관측지점매핑.csv"),
        "model_path": os.path.join(BASE_DIR, "..", "data", "xgboost_best_model.joblib"),
        "output_csv": os.path.join(BASE_DIR, "..", "실행 결과", "xgboost_api_input.csv")
    },
    "constants": {
        "P0": 101325.0,  # 해수면 표준 기압 (Pa)
        "T0": 288.15,    # 해수면 표준 기온 (K)
        "g": 9.80665,    # 중력 가속도 (m/s^2)
        "L": 0.0065,     # 표준 기온 감률 (K/m)
        "R": 8.31447,    # 이상 기체 상수 (J/(mol·K))
        "M": 0.0289644,  # 건조 공기의 몰질량 (kg/mol)
        "Rd": 287.058,   # 건조 공기 기체 상수 (J/(kg·K))
        "Rv": 461.495,   # 수증기 기체 상수 (J/(kg·K))
    }
}

# ==============================================================================
# 계산 함수 (Calculation Functions)
# ==============================================================================

def calculate_pressure_from_altitude(altitude_m: float) -> float:
    """해발고도(m)로 기압(Pa) 계산"""
    c = CONFIG["constants"]
    pressure_pa = c['P0'] * (1 - (c['L'] * altitude_m) / c['T0'])**(c['g'] * c['M'] / (c['R'] * c['L']))
    return pressure_pa

def calculate_vapor_pressure(temp_celsius: float, rh_percent: float) -> tuple[float, float]:
    """기온(°C)과 상대습도(%)로 포화/실제 증기압(Pa) 계산"""
    es_hpa = 6.1121 * math.exp((18.678 - temp_celsius / 234.5) * (temp_celsius / (257.14 + temp_celsius)))
    e_hpa = es_hpa * (rh_percent / 100.0)
    return es_hpa * 100, e_hpa * 100

def calculate_dew_point(temp_celsius: float, rh_percent: float) -> float:
    """기온(°C)과 상대습도(%)로 이슬점온도(°C) 계산"""
    b, c = 17.625, 243.04
    gamma = math.log(rh_percent / 100.0) + (b * temp_celsius) / (c + temp_celsius)
    return (c * gamma) / (b - gamma)

def calculate_air_density(pressure_pa: float, temp_celsius: float, vapor_pressure_pa: float) -> float:
    """기압, 기온, 증기압으로 공기 밀도(kg/m³) 계산"""
    c = CONFIG["constants"]
    temp_kelvin = temp_celsius + 273.15
    dry_air_pressure_pa = pressure_pa - vapor_pressure_pa
    return (dry_air_pressure_pa / (c['Rd'] * temp_kelvin)) + (vapor_pressure_pa / (c['Rv'] * temp_kelvin))

def calculate_absolute_humidity(vapor_pressure_pa: float, temp_celsius: float) -> float:
    """증기압과 기온으로 절대 습도(g/m³) 계산"""
    c = CONFIG["constants"]
    temp_kelvin = temp_celsius + 273.15
    return (vapor_pressure_pa / (c['Rv'] * temp_kelvin)) * 1000

def calculate_turbine_age(base_date: str, built_date: str) -> float:
    """기준일과 준공일로 터빈 연식(년) 계산"""
    base_dt = datetime.strptime(base_date, '%Y%m%d')
    built_dt = datetime.strptime(built_date, '%Y%m%d')
    return round((base_dt - built_dt).days / 365.25, 2)

# ==============================================================================
# 핵심 로직 함수 (Core Logic Functions)
# ==============================================================================

def get_latest_base_time():
    """현재 시간을 기반으로 가장 최신의 base_date와 base_time을 반환합니다."""
    now = datetime.now()
    available_times = ['02', '05', '08', '11', '14', '17', '20', '23']
    current_hour = now.hour
    current_date = now.strftime('%Y%m%d')
    latest_time = None
    for time_str in reversed(available_times):
        if current_hour >= int(time_str):
            latest_time = time_str + '00'
            break
    if latest_time is None:
        yesterday = now - pd.Timedelta(days=1)
        current_date = yesterday.strftime('%Y%m%d')
        latest_time = '2300'
    print(f"현재 시간: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"선택된 기준일시: {current_date} {latest_time}")
    return current_date, latest_time

def get_user_inputs():
    """사용자로부터 예측에 필요한 값을 입력받습니다."""
    print("--- 예측 정보 입력 ---")
    print("예보일시는 현재 시간 기준으로 자동 설정됩니다.")
    
    base_date, base_time = get_latest_base_time()

    try:
        grid_df = pd.read_csv(CONFIG['files']['grid_mapping_csv'])
    except FileNotFoundError:
        print(f"오류: 그리드 매핑 파일 '{CONFIG['files']['grid_mapping_csv']}'을 찾을 수 없습니다.")
        exit()

    step1_list = grid_df['1단계'].dropna().unique().tolist()
    print("\n[1단계] 선택:")
    for i, s1 in enumerate(step1_list): print(f"{i+1}. {s1}")
    try:
        s1_raw = input(f"1단계 번호 입력(1~{len(step1_list)}): ")
        if not s1_raw: print("입력이 없어 프로그램을 종료합니다."); exit()
        s1_val = step1_list[int(s1_raw) - 1]
    except (ValueError, IndexError): print("부적절한 값이 입력되어 프로그램을 종료합니다."); exit()

    step2_list = grid_df[grid_df['1단계'] == s1_val]['2단계'].dropna().unique().tolist()
    if step2_list:
        print("[2단계] 선택:")
        for i, s2 in enumerate(step2_list): print(f"{i+1}. {s2}")
        try:
            s2_raw = input(f"2단계 번호 입력(1~{len(step2_list)}): ")
            if not s2_raw: print("입력이 없어 프로그램을 종료합니다."); exit()
            s2_val = step2_list[int(s2_raw) - 1]
        except (ValueError, IndexError): print("부적절한 값이 입력되어 프로그램을 종료합니다."); exit()
    else: s2_val = None

    if s2_val: step3_list = grid_df[(grid_df['1단계'] == s1_val) & (grid_df['2단계'] == s2_val)]['3단계'].dropna().unique().tolist()
    else: step3_list = grid_df[grid_df['1단계'] == s1_val]['3단계'].dropna().unique().tolist()
    if step3_list:
        print("[3단계] 선택:")
        for i, s3 in enumerate(step3_list): print(f"{i+1}. {s3}")
        try:
            s3_raw = input(f"3단계 번호 입력(1~{len(step3_list)}): ")
            if not s3_raw: print("입력이 없어 프로그램을 종료합니다."); exit()
            s3_val = step3_list[int(s3_raw) - 1]
        except (ValueError, IndexError): print("부적절한 값이 입력되어 프로그램을 종료합니다."); exit()
    else: s3_val = None

    cond = (grid_df['1단계'] == s1_val)
    if s2_val: cond &= (grid_df['2단계'] == s2_val)
    if s3_val: cond &= (grid_df['3단계'] == s3_val)
    row = grid_df[cond]
    if row.empty: 
        print("❌ 해당 행정구역의 좌표 정보를 찾을 수 없습니다.")
        exit()
    else: 
        nx, ny = int(row.iloc[0]['격자 X']), int(row.iloc[0]['격자 Y'])

    # 설비 정보 입력 (기본값 없이 필수 입력)
    try:
        capacity_input = input('🔧 설비용량(MW)을 입력하세요: ')
        if not capacity_input:
            print("❌ 설비용량을 입력해야 합니다.")
            exit()
        capacity = float(capacity_input)
        
        built_date = input("🏗️ 준공일자를 입력하세요 (YYYYMMDD 형식): ")
        if not built_date:
            print("❌ 준공일자를 입력해야 합니다.")
            exit()
            
        blade_input = input('🌪️ 블레이드 길이(m)를 입력하세요: ')
        if not blade_input:
            print("❌ 블레이드 길이를 입력해야 합니다.")
            exit()
        blade_length = float(blade_input)
        
    except ValueError:
        print("❌ 숫자 형식이 올바르지 않습니다.")
        exit()

    inputs = {
        'base_date': base_date, 'base_time': base_time, 'nx': nx, 'ny': ny,
        '설비용량_MW': capacity,
        '준공일': built_date,
        '블레이드': blade_length,
    }
    inputs['연식_년'] = calculate_turbine_age(inputs['base_date'], inputs['준공일'])
    print("-" * 20)
    return inputs


def get_weather_forecast(params: dict):
    """기상청 API를 호출하여 예보 데이터를 가져옵니다."""
    api_config = CONFIG['api']
    api_params = {'serviceKey': api_config['service_key'], 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'JSON', **params}
    try:
        response = requests.get(api_config['base_url'], params=api_params, timeout=30)
        response.raise_for_status()
        print("API 호출 성공!")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 호출 실패: {e}")
        return None

def get_altitude(nx: int, ny: int) -> float:
    """격자 좌표에 해당하는 해발고도를 CSV 파일에서 조회합니다."""
    try:
        grid_df = pd.read_csv(CONFIG['files']['grid_mapping_csv'])
        row = grid_df[(grid_df['격자 X'] == nx) & (grid_df['격자 Y'] == ny)]
        if not row.empty:
            altitude = float(row['노장해발고도(m)'].iloc[0])
            print(f"격자 ({nx}, {ny})의 고도: {altitude:.1f}m")
            return altitude
        else:
            print(f"❌ 격자 ({nx}, {ny})의 고도 정보를 찾을 수 없습니다.")
            exit()
    except (FileNotFoundError, Exception) as e:
        print(f"❌ 고도 정보 로드 오류: {e}")
        exit()

def process_weather_data(api_response: dict, altitude_m: float) -> pd.DataFrame:
    """API 응답을 파싱하고 파생 변수를 추가하여 최종 데이터프레임을 생성합니다."""
    if not api_response or 'response' not in api_response or api_response['response']['header']['resultCode'] != '00':
        print("유효하지 않은 API 응답 데이터입니다.")
        return pd.DataFrame()

    items = api_response['response']['body']['items']['item']
    df = pd.DataFrame(items)
    df['fcstValue'] = pd.to_numeric(df['fcstValue'], errors='coerce')
    pivot_df = df.pivot_table(index=['fcstDate', 'fcstTime'], columns='category', values='fcstValue').reset_index()
    pivot_df['datetime'] = pd.to_datetime(pivot_df['fcstDate'] + pivot_df['fcstTime'], format='%Y%m%d%H%M')

    pressure_pa = calculate_pressure_from_altitude(altitude_m)
    pivot_df['현지기압_hPa'] = pressure_pa / 100

    derived_data = {'증기압_hPa': [], '이슬점온도_°C': [], 'air_density': [], 'absolute_humidity': []}
    for _, row in pivot_df.iterrows():
        temp, humidity = row['TMP'], row['REH']
        _, e_pa = calculate_vapor_pressure(temp, humidity)
        derived_data['증기압_hPa'].append(e_pa / 100)
        derived_data['이슬점온도_°C'].append(calculate_dew_point(temp, humidity))
        derived_data['air_density'].append(calculate_air_density(pressure_pa, temp, e_pa))
        derived_data['absolute_humidity'].append(calculate_absolute_humidity(e_pa, temp))
    for col, data in derived_data.items(): pivot_df[col] = data

    col_rename = {'TMP': '기온(°C)', 'WSD': '풍속(m/s)', 'REH': '습도(%)', 'VEC': '풍향(deg)', 'PCP': '강수량(mm)'}
    pivot_df.rename(columns=col_rename, inplace=True)
    
    if '강수량(mm)' in pivot_df.columns:
        pivot_df['강수량(mm)'] = pd.to_numeric(pivot_df['강수량(mm)'], errors='coerce').fillna(0).clip(lower=0)
    else:
        pivot_df['강수량(mm)'] = 0
    
    print("데이터 처리 및 파생 변수 생성 완료.")
    return pivot_df

def create_model_features(df: pd.DataFrame, turbine_specs: dict) -> pd.DataFrame:
    """모델 예측에 사용할 최종 피처를 생성합니다."""
    features_df = df.copy()
    dt = features_df['datetime'].dt
    features_df['시간_sin'] = np.sin(2 * np.pi * dt.hour / 24)
    features_df['시간_cos'] = np.cos(2 * np.pi * dt.hour / 24)
    features_df['월_sin'] = np.sin(2 * np.pi * (dt.month - 1) / 12)
    features_df['월_cos'] = np.cos(2 * np.pi * (dt.month - 1) / 12)
    features_df['풍속(m/s)_cubed'] = features_df['풍속(m/s)'] ** 3
    wind_dir_rad = np.deg2rad(features_df['풍향(deg)'])
    features_df['풍향_sin'] = np.sin(wind_dir_rad)
    features_df['풍향_cos'] = np.cos(wind_dir_rad)
    features_df['회전체면적'] = turbine_specs['블레이드'] ** 2 * np.pi
    features_df['설비용량(MW)'] = turbine_specs['설비용량(MW)']
    features_df['연식(년)'] = turbine_specs['연식(년)']

    final_cols_map = {'증기압_hPa': '증기압(hPa)', '이슬점온도_°C': '이슬점온도(°C)', '현지기압_hPa': '현지기압(hPa)'}
    features_df.rename(columns=final_cols_map, inplace=True)

    model_features_order = [
        '설비용량(MW)', '연식(년)', '기온(°C)', '풍속(m/s)_cubed', '습도(%)',
        '증기압(hPa)', '이슬점온도(°C)', '현지기압(hPa)', '풍향_sin', '풍향_cos',
        '시간_sin', '시간_cos', '월_sin', '월_cos', '회전체면적',
        'air_density', 'absolute_humidity', '강수량(mm)'
    ]
    return features_df[model_features_order]

def calculate_daily_totals(result_df: pd.DataFrame) -> pd.DataFrame:
    """일별 총 발전량을 계산합니다."""
    result_df['date_obj'] = pd.to_datetime(result_df['date'])
    daily_totals = result_df.groupby('date')['prediction_kWh'].sum().reset_index()
    daily_totals['day_name'] = pd.to_datetime(daily_totals['date']).dt.strftime('%Y-%m-%d (%A)')
    
    # 오늘, 내일, 모레 구분
    today = datetime.now().date()
    daily_totals['day_type'] = daily_totals['date'].apply(lambda x: 
        '오늘' if pd.to_datetime(x).date() == today else
        '내일' if pd.to_datetime(x).date() == today + pd.Timedelta(days=1) else
        '모레' if pd.to_datetime(x).date() == today + pd.Timedelta(days=2) else
        '기타'
    )
    return daily_totals

def print_daily_totals(daily_totals: pd.DataFrame):
    """일별 총 발전량 중 내일/모레만 터미널에 출력합니다."""
    print("\n" + "="*50)
    print("📊 일별 총 발전량 예측 (내일/모레만)")
    print("="*50)
    
    for _, row in daily_totals.iterrows():
        day_type = row['day_type']
        total_kwh = row['prediction_kWh']
        date_str = row['day_name']
        
        if day_type in ['내일', '모레']:
            print(f"🔹 {day_type} ({date_str}): {total_kwh:,.2f} kWh")
    
    print("="*50)

def show_hourly_generation_graph(result_df: pd.DataFrame):
    """시간별 발전량 꺾은선그래프를 출력합니다."""
    try:
        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        
        # 데이터 준비
        result_df['datetime'] = pd.to_datetime(result_df['date'] + ' ' + result_df['time'])
        result_df = result_df.sort_values('datetime')
        
        # 오늘, 내일, 모레 데이터만 필터링
        today = datetime.now().date()
        target_dates = [today, today + pd.Timedelta(days=1), today + pd.Timedelta(days=2)]
        filtered_df = result_df[result_df['datetime'].dt.date.isin(target_dates)]
        
        if filtered_df.empty:
            print("⚠️ 오늘/내일/모레 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        
        for i, target_date in enumerate(target_dates):
            day_data = filtered_df[filtered_df['datetime'].dt.date == target_date]
            if not day_data.empty:
                day_name = ['오늘', '내일', '모레'][i]
                plt.plot(day_data['datetime'], day_data['prediction_kWh'], 
                        marker='o', linewidth=2, markersize=4, label=f'{day_name} ({target_date})')
        
        plt.title('시간별 풍력 발전량 예측', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('시간', fontsize=12)
        plt.ylabel('발전량 (kWh)', fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        print("\n📈 시간별 발전량 그래프를 표시합니다...")
        plt.show()
        
    except Exception as e:
        print(f"⚠️ 그래프 생성 중 오류 발생: {e}")
        print("matplotlib 설치가 필요할 수 있습니다: pip install matplotlib")
    
# ==============================================================================
# 메인 실행 블록 (Main Execution Block)
# ==============================================================================

def main():
    """메인 실행 함수"""
    user_inputs = get_user_inputs()
    api_params = {'base_date': user_inputs['base_date'], 'base_time': user_inputs['base_time'], 'nx': str(user_inputs['nx']), 'ny': str(user_inputs['ny'])}
    turbine_specs = {'설비용량(MW)': user_inputs['설비용량_MW'], '연식(년)': user_inputs['연식_년'], '블레이드': user_inputs['블레이드']}

    weather_data = get_weather_forecast(api_params)
    if not weather_data: return

    altitude = get_altitude(user_inputs['nx'], user_inputs['ny'])
    processed_df = process_weather_data(weather_data, altitude)
    if processed_df.empty: return

    model_input_df = create_model_features(processed_df, turbine_specs)
    
    output_path = CONFIG['files']['output_csv']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model_input_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"모델 입력 데이터가 '{output_path}' 파일로 저장되었습니다.")

    # 7. 모델 로드 및 예측
    try:
        model = joblib.load(CONFIG['files']['model_path'])
        
        # 모델의 '날것(raw)' 예측값을 받습니다.
        # 이전 학습 코드에 따르면 모델은 로그 변환 없이 원본 스케일로 학습되었습니다.
        # 따라서 np.expm1()은 사용하지 않아야 합니다.
        raw_predictions = model.predict(model_input_df)

        # ⭐️ 중요: 예측된 음수 값을 0으로 처리하는 후처리 로직 적용
        predictions = np.clip(raw_predictions, 0, None)

        print("\n--- 예측 결과 ---")
        print(f"총 {len(predictions)}개의 시점에 대한 발전량 예측 완료.")
        print("예측값 샘플 (처음 10개):")
        print(np.round(predictions[:10], 2))

        processed_df['예측발전량(kWh)'] = predictions
        print("\n=== 시간대별 예측 결과 (상위 5개) ===")
        print(processed_df[['datetime', '기온(°C)', '풍속(m/s)', '예측발전량(kWh)']].head())

        prediction_output_path = os.path.abspath(
            os.path.join(BASE_DIR, "..", "..", "..", "최종_발전량_예측_모델", "result", "wind_prediction_results.csv")
        )
        os.makedirs(os.path.dirname(prediction_output_path), exist_ok=True)
        result_df = processed_df[['datetime', '예측발전량(kWh)', '기온(°C)', '풍속(m/s)', '풍향(deg)']].copy()
        result_df['date'] = result_df['datetime'].dt.strftime('%Y-%m-%d')
        result_df['time'] = result_df['datetime'].dt.strftime('%H:%M')
        result_df = result_df[['date', 'time', '기온(°C)', '풍속(m/s)', '풍향(deg)', '예측발전량(kWh)']]
        result_df.rename(columns={'예측발전량(kWh)': 'prediction_kWh', '기온(°C)': 'temperature_C', '풍속(m/s)': 'wind_speed_ms', '풍향(deg)': 'wind_direction_deg'}, inplace=True)
        result_df.to_csv(prediction_output_path, index=False, encoding='utf-8-sig')
        print(f"예측 결과가 '{prediction_output_path}' 파일로 저장되었습니다.")

        # 일별 총 발전량 계산 및 출력
        daily_totals = calculate_daily_totals(result_df)
        print_daily_totals(daily_totals)
        
        # 시간별 발전량 그래프 출력
        show_hourly_generation_graph(result_df)

    except FileNotFoundError:
        print(f"모델 파일 '{CONFIG['files']['model_path']}'을 찾을 수 없습니다.")
    except Exception as e:
        print(f"모델 예측 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
