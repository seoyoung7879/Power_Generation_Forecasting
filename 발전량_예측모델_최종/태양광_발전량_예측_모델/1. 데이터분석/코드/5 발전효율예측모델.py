import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import KFold

df = pd.read_csv("../최종사용_데이터/발전효율모델_최종데이터.csv", encoding='utf-8')

# ✅ 원하는 컬럼을 아래 리스트에서 선택하세요
selected_cols = [
    '연식(년)',
    'month_day_sin', 'month_day_cos', 'hour_sin', 'hour_cos',
    '태양고도', '방위각', '일사(MJ/m2)', '기온(°C)', '습도(%)', '이슬점', 'T-Td',
    '강수량(mm)', '하늘상태', '풍속(m/s)'
]

def run_two_stage_model_logclip(df, selected_cols, target_col='발전효율', n_trials=50):
    df_model = df.dropna(subset=[target_col]).copy()
    X = df_model[selected_cols]
    y = df_model[target_col]

    # ✅ 분류용 타겟 생성: 발전 안 한 날은 1, 한 날은 0
    y_cls = (y == 0).astype(int)

    # ✅ 학습/검증 분할
    X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)
    X_train_reg = X[y_cls == 0]
    y_train_reg = y[y_cls == 0]
    X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(X_train_reg, y_train_reg, test_size=0.2, random_state=42)

    # ✅ log1p 변환
    y_train_reg_log = np.log1p(y_train_reg)
    y_val_reg_log = np.log1p(y_val_reg)

    # ✅ 분류 모델 튜닝 with KFold
    def objective_cls(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 500),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 1.0, 5.0),
            "verbose": 0,
            "task_type": "GPU",
            "random_state": 42
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accs = []
        for train_idx, val_idx in kf.split(X_train_cls):
            model = CatBoostClassifier(**params)
            model.fit(X_train_cls.iloc[train_idx], y_train_cls.iloc[train_idx])
            pred = model.predict(X_train_cls.iloc[val_idx])
            acc = accuracy_score(y_train_cls.iloc[val_idx], pred)
            accs.append(acc)
        mean_acc = np.mean(accs)
        print(f"🔍 분류 Trial | Mean Accuracy (CV): {mean_acc:.4f}")
        return 1 - mean_acc

    study_cls = optuna.create_study(direction="minimize")
    study_cls.optimize(objective_cls, n_trials=n_trials)

    best_params_cls = study_cls.best_trial.params
    final_cls = CatBoostClassifier(**best_params_cls, verbose=0, task_type="GPU", random_state=42)
    final_cls.fit(X_train_cls, y_train_cls)

    # ✅ 회귀 모델 튜닝 with KFold
    def objective_reg(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
            "random_strength": trial.suggest_float("random_strength", 1.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "loss_function": "Poisson",
            "verbose": 0,
            "task_type": "GPU",
            "random_state": 42
        }
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rmses = []
        for train_idx, val_idx in kf.split(X_train_reg):
            model = CatBoostRegressor(**params)
            model.fit(X_train_reg.iloc[train_idx], y_train_reg_log.iloc[train_idx])
            pred_log = model.predict(X_train_reg.iloc[val_idx])
            pred = np.clip(np.expm1(pred_log), 0, None)
            rmse = np.sqrt(mean_squared_error(y_train_reg.iloc[val_idx], pred))
            rmses.append(rmse)
        mean_rmse = np.mean(rmses)
        print(f"📈 회귀 Trial | Mean RMSE (CV): {mean_rmse:.4f}")
        return mean_rmse

    study_reg = optuna.create_study(direction="minimize")
    study_reg.optimize(objective_reg, n_trials=n_trials)

    best_params_reg = study_reg.best_trial.params
    final_reg = CatBoostRegressor(**best_params_reg, verbose=0, task_type="GPU", random_state=42)
    final_reg.fit(X_train_reg, y_train_reg_log)

    # ✅ 전체 예측
    df_all = df_model.copy()
    X_all = df_all[selected_cols]
    y_true = df_all[target_col].values

    is_zero_pred = final_cls.predict(X_all)
    eff_pred = np.zeros(len(df_all))
    idx_nonzero = np.where(is_zero_pred == 0)[0]

    pred_log = final_reg.predict(X_all.iloc[idx_nonzero])
    pred_real = np.clip(np.expm1(pred_log), 0, None)
    eff_pred[idx_nonzero] = pred_real

    # ✅ 성능 출력
    mse = mean_squared_error(y_true, eff_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, eff_pred)
    r2 = r2_score(y_true, eff_pred)
    smape = 100 * np.mean(2 * np.abs(y_true - eff_pred) / (np.abs(y_true) + np.abs(eff_pred) + 1e-6))

    print("\n✅ 최종 Two-Stage(log+clip+Poisson) 모델 성능:")
    print(f"  🔹 RMSE  : {rmse:.4f}")
    print(f"  🔹 MAE   : {mae:.4f}")
    print(f"  🔹 R²    : {r2:.4f}")
    print(f"  🔹 SMAPE : {smape:.2f}%")

    df_all['예측효율'] = eff_pred
    return final_cls, final_reg, df_all


final_cls, final_reg, df_result = run_two_stage_model_logclip(df, selected_cols)