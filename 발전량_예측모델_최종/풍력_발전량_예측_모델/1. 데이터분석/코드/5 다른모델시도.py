import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import optuna
import warnings

# --- 1. 초기 설정 ---
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING) # 로그를 간결하게

RANDOM_STATE = 42    # 결과 재현을 위한 시드값

# --- 2. 데이터 준비 ---
print("데이터 로드를 시작합니다...")
df = pd.read_csv('풍력_공모전/풍력_발전량_예측_모델/1. 데이터분석/최종사용_데이터/파생변수추가_기상과풍력.csv')

# 전처리
if df.isnull().sum().sum() > 0: df = df.dropna()
if '풍속(m/s)' in df.columns: df['풍속(m/s)_cubed'] = df['풍속(m/s)'] ** 3
if '블레이드' in df.columns: df['회전체면적'] = (df['블레이드'] ** 2) * np.pi

# 특성(X) 및 타겟(y) 설정
target = '발전량(kWh)'
features = [
    '설비용량(MW)', '연식(년)', '기온(°C)', '풍속(m/s)_cubed', '습도(%)', '증기압(hPa)', '이슬점온도(°C)',
    '현지기압(hPa)', '풍향_sin', '풍향_cos', '시간_sin', '시간_cos', '월_sin', '월_cos', '회전체면적',
    'air_density', 'absolute_humidity', '강수량(mm)'
]
available_features = [col for col in features if col in df.columns]
X = df[available_features]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
print(f"데이터 준비 완료. 학습: {len(X_train)}개, 검증: {len(X_val)}개\n")

# --- 3. 모델별 실험 함수 ---
def run_experiment(model_name):
    print(f"[{model_name}] 모델 처리를 시작합니다...")
    print("  > 미리 정의된 최적 파라미터를 사용합니다.")

    # 모델별로 미리 찾은 최적 파라미터 정의
    if model_name == 'RandomForest':
        params = {'n_estimators': 854, 'max_depth': 40, 'min_samples_split': 2, 'min_samples_leaf': 1}
        model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=-1)

    elif model_name == 'XGBoost':
        params = {
            'n_estimators': 2872, 'learning_rate': 0.0335762440419045, 'max_depth': 10,
            'subsample': 0.7272212627300844, 'colsample_bytree': 0.9254306650281618
        }
        model = XGBRegressor(**params, tree_method='gpu_hist', predictor='gpu_predictor', random_state=RANDOM_STATE, n_jobs=-1)

    elif model_name == 'LightGBM':
        params = {
            'n_estimators': 2717, 'learning_rate': 0.09759070734303117,
            'num_leaves': 142, 'max_depth': 15
        }
        model = LGBMRegressor(**params, device='gpu', random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)

    elif model_name == 'CatBoost':
        params = {
            'iterations': 3929,
            'learning_rate': 0.08808498552768322,
            'depth': 12,
            'l2_leaf_reg': 1.8083641221733064,
            'bootstrap_type': 'MVS',
            'loss_function': 'MAE',
            'random_seed': 42,
            'verbose': 0,
            'task_type': 'CPU'
        }
        model = CatBoostRegressor(**params)

    # 모델 학습
    model.fit(X_train, y_train)

    # --- (참고용 주석) 이전에 사용했던 전체 튜닝 코드 ---
    """
    def objective_template(trial):
        # RandomForest
        rf_params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000), 'max_depth': trial.suggest_int('max_depth', 10, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 32), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 32),
        }
        # XGBoost
        xgb_params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 5, 15),
        }
        # LightGBM
        lgbm_params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 3000), 'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150), 'max_depth': trial.suggest_int('max_depth', 5, 15),
        }
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_template, n_trials=100)
    """
    # --- (참고용 주석) CatBoost 튜닝 코드 ---
    # from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'iterations': [400, 600, 800, 1000],
#     'learning_rate': [0.05, 0.07, 0.1],
#     'depth': [6, 8, 10],
#     'l2_leaf_reg': [1, 3, 5],
#     'bootstrap_type': ['Bayesian', 'Bernoulli']
# }
# catboost_model = CatBoostRegressor(loss_function='RMSE', random_seed=42, verbose=0)
# grid_search = GridSearchCV(
#     catboost_model,
#     param_grid,
#     cv=3,
#     scoring='neg_root_mean_squared_error',
#     n_jobs=-1,
#     verbose=2
# )
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# print(f"   최적 파라미터: {best_params}")
# print(f"   최적 CV 점수: {-grid_search.best_score_:.2f}")

    # --- 4. 최종 평가 ---
    y_pred = model.predict(X_val)
    y_pred = np.clip(y_pred, 0, None)  # 음수를 0으로 클리핑

    # 모든 성능 지표 계산
    metrics = {'Model': model_name}
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)
    y_range = y_val.max() - y_val.min()
    nmae = (mae / y_range) * 100 if y_range > 0 else 0
    nrmse = (rmse / y_range) * 100 if y_range > 0 else 0
    mbe = np.mean(y_pred - y_val)

    eff_mae, eff_rmse, eff_median = np.nan, np.nan, np.nan
    if '설비용량(MW)' in X_val.columns:
        efficiency_true = y_val / X_val['설비용량(MW)']
        efficiency_pred = y_pred / X_val['설비용량(MW)']  # 이미 클리핑된 y_pred 사용
        eff_mae = mean_absolute_error(efficiency_true, efficiency_pred)
        eff_rmse = np.sqrt(mean_squared_error(efficiency_true, efficiency_pred))
        eff_median = np.median(np.abs(efficiency_true - efficiency_pred))

    # 음수 예측 개수 확인 (클리핑 전 원본 예측값에서)
    y_pred_original = model.predict(X_val)
    neg_count = np.sum(y_pred_original < 0)
    neg_ratio = (neg_count / len(y_pred_original)) * 100

    print(f"  > 음수 예측: {neg_count}개 ({neg_ratio:.2f}%) → 0으로 처리됨")

    metrics.update({
        'MAE': mae, 'RMSE': rmse, 'R² Score': r2, 'nMAE (%)': nmae,
        'nRMSE (%)': nrmse, 'MBE': mbe, 'Eff. MAE': eff_mae,
        'Eff. RMSE': eff_rmse, 'Eff. Median Err': eff_median,
        'Negative Predictions': neg_count  # 음수 예측 개수도 결과에 포함
    })
    return metrics

# --- 5. 메인 실행 로직 ---
models_to_run = ['RandomForest', 'XGBoost', 'LightGBM', 'CatBoost']
results_list = []

for model in models_to_run:
    result = run_experiment(model)
    results_list.append(result)
    print("-" * 80)

# 최종 결과 종합 및 출력
results_df = pd.DataFrame(results_list).sort_values(by='RMSE', ascending=True)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.width', 120)

print("\n\n📊 [최종 결과] 모델별 상세 성능 비교 (RMSE 낮은 순 정렬)")
print(results_df.to_string(index=False))
