# -*- coding: utf-8 -*-
import math
from datetime import datetime, timedelta
import pandas as pd
import requests
import joblib
import numpy as np
import pytz
import pvlib
import calendar
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os
# ==============================================================================
# 설정 (Configurations)
# ==============================================================================
# 경로 설정 (공모전용 상대경로)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG = {
    "api": {
        "service_key": "zJLFmDMckurk+au32kOHTxsrU5gG2NAadNE68xYaBW8PBJtdXN7F4QEpuW6f68GL0qLcMQsmgyPHxbOs43NCBA==",
        "base_url": "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
    },
    "files": {
        "grid_mapping_csv": os.path.join(BASE_DIR, "..", "data", "격자예보_관측지점매핑.csv"),
        "insolation_model_path": os.path.join(BASE_DIR, "..", "data", "일사모델.joblib"),
        "power_model_path": os.path.join(BASE_DIR, "..", "data", "태양광_발전량_모델.joblib"),
        "output_csv": os.path.join(BASE_DIR, "..", "실행 결과", "solar_prediction_results_final.csv")
    }
}

# ==============================================================================
# 계산 함수 (Calculation Functions)
# ==============================================================================
def calculate_age(base_date_str: str, built_date_str: str) -> float:
    """기준일과 준공일로 설비 연식(년) 계산"""
    try:
        base_date = datetime.strptime(base_date_str, '%Y%m%d')
        built_date = datetime.strptime(built_date_str, '%Y%m%d')
        return round((base_date - built_date).days / 365.25, 2)
    except ValueError:
        print(f"❌ 잘못된 날짜 형식입니다: {base_date_str} 또는 {built_date_str}. YYYYMMDD 형식을 사용하세요.")
        exit()

def calculate_dew_point(temp_celsius: float, rh_percent: float) -> float:
    """기온(°C)과 상대습도(%)로 이슬점온도(°C) 계산"""
    b, c = 17.62, 243.12
    rh_percent = max(rh_percent, 0.1)
    gamma = (b * temp_celsius / (c + temp_celsius)) + np.log(rh_percent / 100.0)
    return (c * gamma) / (b - gamma)

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

def get_user_inputs(base_date: str) -> dict:
    """사용자로부터 예측에 필요한 정보를 입력받습니다."""
    print("--- 태양광 발전량 예측 정보 입력 ---")
    try:
        grid_df = pd.read_csv(CONFIG['files']['grid_mapping_csv'])
    except FileNotFoundError:
        print(f"❌ 오류: 매핑 파일 '{CONFIG['files']['grid_mapping_csv']}'을 찾을 수 없습니다.")
        exit()

    # 주소 선택 로직
    step1_list = grid_df['1단계'].dropna().unique().tolist()
    print("\n[1단계] 시/도 선택:")
    for i, s1 in enumerate(step1_list): print(f"{i+1}. {s1}")
    try:
        s1_idx = int(input(f"→ 1단계 번호 입력(1~{len(step1_list)}): ")) - 1
        s1_val = step1_list[s1_idx]
        step2_list = grid_df[grid_df['1단계'] == s1_val]['2단계'].dropna().unique().tolist()
        print("\n[2단계] 시/군/구 선택:")
        for i, s2 in enumerate(step2_list): print(f"{i+1}. {s2}")
        s2_idx = int(input(f"→ 2단계 번호 입력(1~{len(step2_list)}): ")) - 1
        s2_val = step2_list[s2_idx]
        step3_list = grid_df[(grid_df['1단계'] == s1_val) & (grid_df['2단계'] == s2_val)]['3단계'].dropna().unique().tolist()
        print("\n[3단계] 읍/면/동 선택:")
        for i, s3 in enumerate(step3_list): print(f"{i+1}. {s3}")
        s3_idx = int(input(f"→ 3단계 번호 입력(1~{len(step3_list)}): ")) - 1
        s3_val = step3_list[s3_idx]
    except (ValueError, IndexError):
        print("❌ 부적절한 값이 입력되어 프로그램을 종료합니다.")
        exit()
    
    row = grid_df[(grid_df['1단계'] == s1_val) & (grid_df['2단계'] == s2_val) & (grid_df['3단계'] == s3_val)]
    if row.empty:
        print("❌ 해당 행정구역의 좌표 정보를 찾을 수 없습니다.")
        exit()
    
    # 설비 정보 입력
    try:
        capacity = float(input("\n🔧 설비용량(MW)을 입력하세요: "))
        built_date = input("🏗️ 준공일자를 입력하세요 (YYYYMMDD 형식): ")
    except ValueError:
        print("❌ 설비용량은 숫자로 입력해야 합니다.")
        exit()

    inputs = {
        'nx': int(row['격자 X'].iloc[0]), 'ny': int(row['격자 Y'].iloc[0]),
        '위도': float(row['위도'].iloc[0]), '경도': float(row['경도'].iloc[0]),
        '고도': float(row['노장해발고도(m)'].iloc[0]), '설비용량(MW)': capacity,
        '연식(년)': calculate_age(base_date, built_date)
    }
    print("-" * 20)
    return inputs

def get_weather_forecast(params: dict) -> list | None:
    """기상청 API를 호출하여 예보 데이터를 가져옵니다."""
    api_config = CONFIG['api']
    api_params = {'serviceKey': api_config['service_key'], 'pageNo': '1', 'numOfRows': '1000', 'dataType': 'JSON', **params}
    try:
        print("📡 기상청 API에서 예보 데이터를 요청합니다...")
        response = requests.get(api_config['base_url'], params=api_params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if data['response']['header']['resultCode'] != '00':
            print(f"❌ API 오류: {data['response']['header']['resultMsg']}")
            return None
        print("✅ API 호출 성공!")
        return data['response']['body']['items'].get('item', [])
    except requests.exceptions.RequestException as e:
        print(f"❌ API 호출 실패: {e}")
        return None

def process_weather_data(items: list, location_info: dict) -> pd.DataFrame:
    """API 응답을 파싱하고 모델에 필요한 기본 피처를 생성합니다."""
    if not items: return pd.DataFrame()
    df = pd.DataFrame(items)
    df['fcstValue'] = pd.to_numeric(df['fcstValue'], errors='coerce')
    pivot_df = df.pivot_table(index=['fcstDate', 'fcstTime'], columns='category', values='fcstValue').reset_index()
    rename_dict = {'TMP': '기온(°C)', 'PCP': '강수량(mm)', 'REH': '습도(%)', 'WSD': '풍속(m/s)', 'SNO': '적설(cm)', 'SKY': '하늘상태'}
    pivot_df.rename(columns=rename_dict, inplace=True)
    pivot_df.fillna(0, inplace=True)
    seoul_tz = pytz.timezone('Asia/Seoul')
    pivot_df['datetime'] = pd.to_datetime(pivot_df['fcstDate'] + pivot_df['fcstTime'], format='%Y%m%d%H%M').dt.tz_localize(seoul_tz)
    
    pivot_df['month'] = pivot_df['datetime'].dt.month
    pivot_df['hour'] = pivot_df['datetime'].dt.hour
    pivot_df['month_sin'] = np.sin(2 * np.pi * pivot_df['month'] / 12)
    pivot_df['month_cos'] = np.cos(2 * np.pi * pivot_df['month'] / 12)
    pivot_df['hour_sin'] = np.sin(2 * np.pi * pivot_df['hour'] / 24)
    pivot_df['hour_cos'] = np.cos(2 * np.pi * pivot_df['hour'] / 24)
    pivot_df['days_in_month'] = pivot_df['datetime'].dt.days_in_month
    pivot_df['month_day'] = pivot_df['datetime'].dt.month + (pivot_df['datetime'].dt.day - 1) / pivot_df['days_in_month']
    pivot_df['month_day_sin'] = np.sin(2 * np.pi * pivot_df['month_day'] / 12)
    pivot_df['month_day_cos'] = np.cos(2 * np.pi * pivot_df['month_day'] / 12)
    pivot_df['이슬점'] = pivot_df.apply(lambda row: calculate_dew_point(row['기온(°C)'], row['습도(%)']), axis=1)
    pivot_df['T-Td'] = pivot_df['기온(°C)'] - pivot_df['이슬점']
    
    for key, value in location_info.items(): pivot_df[key] = value
    print("✅ 데이터 처리 및 기본 피처 생성 완료.")
    return pivot_df

def add_solar_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """pvlib를 사용하여 태양고도와 방위각을 계산합니다."""
    print("⏳ 태양고도 및 방위각 계산 중 (pvlib 사용)...")
    loc = pvlib.location.Location(latitude=df['위도'].iloc[0], longitude=df['경도'].iloc[0], tz='Asia/Seoul', altitude=df['고도'].iloc[0])
    solar_positions = loc.get_solarposition(times=df['datetime'])
    df['태양고도'] = solar_positions['apparent_elevation'].values
    df['방위각'] = solar_positions['azimuth'].values
    print("✅ 태양고도 및 방위각 계산 완료.")
    return df

def predict_insolation(df: pd.DataFrame) -> pd.DataFrame:
    """2-Step 모델(분류+회귀)을 사용하여 일사량을 예측합니다."""
    print(f"\n모델 로딩 (일사량 2-Step): '{CONFIG['files']['insolation_model_path']}'")
    try:
        model_dict = joblib.load(CONFIG['files']['insolation_model_path'])
        classifier = model_dict['classifier']
        regressor = model_dict['regressor']
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ 일사량 모델 파일 로딩 오류: {e}. 예측을 중단합니다.")
        return pd.DataFrame()

    # 일사량 모델이 요구하는 피처 목록
    feature_cols = [
        'month_day_sin', 'month_day_cos', 'hour_sin', 'hour_cos',
        '태양고도', '방위각', '기온(°C)', '풍속(m/s)', '습도(%)',
        '강수량(mm)', '하늘상태', 'T-Td'
    ]
    
    input_df = df[feature_cols].copy()
    input_df['하늘상태'] = input_df['하늘상태'].astype('category')

    print("🌤️ 일사량 예측 중 (1/2: 분류 모델)...")
    is_zero_pred = classifier.predict(input_df)

    print("🌤️ 일사량 예측 중 (2/2: 회귀 모델)...")
    final_pred = np.zeros(len(input_df))
    idx_nonzero = np.where(is_zero_pred == 0)[0]
    
    if len(idx_nonzero) > 0:
        # --- 여기를 수정했습니다 ---
        try:
            # 원본 로직: .booster_.predict() 사용을 우선 시도
            pred_real = regressor.booster_.predict(input_df.iloc[idx_nonzero])
        except AttributeError:
            # .booster_ 속성이 없는 경우, 일반 .predict() 사용
            print("   - (정보) .booster_ 속성을 찾을 수 없어 일반 .predict()를 사용합니다.")
            pred_real = regressor.predict(input_df.iloc[idx_nonzero])
        # --- 여기까지 수정 ---
        
        final_pred[idx_nonzero] = np.clip(pred_real, 0, None)

    df['일사(MJ/m2)'] = final_pred
    print("✅ 일사량 예측 완료.")
    return df

def predict_power_generation(df: pd.DataFrame) -> np.ndarray | None:
    """2-Step 모델(분류+회귀)을 사용하여 최종 발전 효율을 예측합니다."""
    print(f"\n모델 로딩 (최종 발전량 2-Step): '{CONFIG['files']['power_model_path']}'")
    try:
        model_dict = joblib.load(CONFIG['files']['power_model_path'])
        classifier = model_dict['classifier']
        regressor = model_dict['regressor']
        feature_cols = model_dict['features']
    except (FileNotFoundError, KeyError) as e:
        print(f"❌ 최종 발전량 모델 파일 로딩 오류: {e}. 예측을 중단합니다.")
        return None

    # 모델에 필요한 피처가 모두 있는지 확인
    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ 최종 모델에 필요한 피처가 부족합니다: {missing_cols}")
        return None

    input_df = df[feature_cols].copy()
    
    # --- 여기를 수정했습니다 ---
    # 최종 모델의 학습 방식에 맞춰 '하늘상태' 컬럼을 정수형(int)으로만 유지합니다.
    # .astype('category') 변환을 제거합니다.
    input_df['하늘상태'] = input_df['하늘상태'].astype('int')
    # --------------------------

    print("🔋 최종 발전량 예측 중 (1/2: 분류)...")
    is_zero_pred = classifier.predict(input_df)
    print("🔋 최종 발전량 예측 중 (2/2: 회귀)...")
    final_pred_efficiency = np.zeros(len(input_df))
    idx_nonzero = np.where(is_zero_pred == 0)[0]

    if len(idx_nonzero) > 0:
        pred_log = regressor.predict(input_df.iloc[idx_nonzero])
        pred_real = np.expm1(pred_log)
        final_pred_efficiency[idx_nonzero] = np.clip(pred_real, 0, None)

    print("✅ 최종 발전량(효율) 예측 완료.")
    return final_pred_efficiency

def calculate_daily_totals(result_df: pd.DataFrame) -> pd.DataFrame:
    """일별 총 발전량을 계산합니다."""
    result_df['date'] = pd.to_datetime(result_df['datetime']).dt.date
    daily_totals = result_df.groupby('date')['예측발전량(kWh)'].sum().reset_index()
    daily_totals['date'] = daily_totals['date'].astype(str)
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
        total_kwh = row['예측발전량(kWh)']
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
        result_df['date'] = pd.to_datetime(result_df['datetime']).dt.date
        result_df = result_df.sort_values('datetime')
        
        # 오늘, 내일, 모레 데이터만 필터링
        today = datetime.now().date()
        target_dates = [today, today + pd.Timedelta(days=1), today + pd.Timedelta(days=2)]
        filtered_df = result_df[result_df['date'].isin(target_dates)]
        
        if filtered_df.empty:
            print("⚠️ 오늘/내일/모레 데이터가 없어 그래프를 생성할 수 없습니다.")
            return
        
        # 그래프 생성
        plt.figure(figsize=(12, 6))
        
        for i, target_date in enumerate(target_dates):
            day_data = filtered_df[filtered_df['date'] == target_date]
            if not day_data.empty:
                day_name = ['오늘', '내일', '모레'][i]
                plt.plot(day_data['datetime'], day_data['예측발전량(kWh)'], 
                        marker='o', linewidth=2, markersize=4, label=f'{day_name} ({target_date})')
        
        plt.title('시간별 태양광 발전량 예측', fontsize=16, fontweight='bold', pad=20)
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
    print("--- 태양광 발전량 예측 시작 ---")
    print("예보일시는 현재 시간 기준으로 자동 설정됩니다.")
    
    base_date, base_time = get_latest_base_time()

    user_inputs = get_user_inputs(base_date)
    api_params = {'base_date': base_date, 'base_time': base_time, 'nx': str(user_inputs['nx']), 'ny': str(user_inputs['ny'])}
    weather_items = get_weather_forecast(api_params)
    if not weather_items: return

    processed_df = process_weather_data(weather_items, user_inputs)
    if processed_df.empty: return

    df_with_solar_pos = add_solar_position_features(processed_df.copy())
    df_with_insolation = predict_insolation(df_with_solar_pos)
    if df_with_insolation.empty: return
    
    predicted_efficiency = predict_power_generation(df_with_insolation)
    if predicted_efficiency is None: return

    capacity_mw = user_inputs['설비용량(MW)']
    predictions_kwh = predicted_efficiency * capacity_mw * 1000

    print("\n--- 🌞 최종 예측 결과 ---")
    
    # --- 여기를 수정했습니다 ---
    # 'Predicted_Solar' 대신, 실제로 존재하는 컬럼명인 '일사(MJ/m2)'를 사용합니다.
    result_df = df_with_insolation[['datetime', '기온(°C)', '습도(%)', '태양고도', '일사(MJ/m2)']].copy()
    result_df.rename(columns={'일사(MJ/m2)': '예측일사량(MJ/m2)'}, inplace=True)
    # --------------------------

    result_df['예측효율'] = np.round(predicted_efficiency, 4)
    result_df['예측발전량(kWh)'] = np.round(predictions_kwh, 2)
    
    # 최종 결과를 통합된 result 폴더에 저장
    output_path = os.path.abspath(
        os.path.join(BASE_DIR, "..", "..", "..", "최종_발전량_예측_모델", "result", "solar_prediction_results.csv")
    )
    # 결과 디렉터리가 없으면 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result_df['datetime'] = result_df['datetime'].dt.tz_localize(None)
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 최종 예측 결과가 '{output_path}' 파일로 저장되었습니다.")
    print("\n=== 시간대별 예측 결과 (상위 10개) ===")
    print(result_df.head(10).to_string())

    # 일별 총 발전량 계산 및 출력
    daily_totals = calculate_daily_totals(result_df)
    print_daily_totals(daily_totals)
    
    # 시간별 발전량 그래프 출력
    show_hourly_generation_graph(result_df)


if __name__ == "__main__":
    main()
