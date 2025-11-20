#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RF + SMOTE(k_neighbors=1) + StratifiedKFold + Optuna
同时输出：
 - RF 内建 feature_importances_ 的前150特征
 - Permutation Importance 的前150特征

说明：
 - 使用 DataFrame 格式保留特征名
 - 在 Optuna CV 中对 train 部分使用 SMOTE（避免泄漏）
 - 最终训练：用 SMOTE 对全量 217 样本训练用于预测
 - 另外训练一个不含 SMOTE 的 RF（在同样 scaler 下）用于计算特征重要性（Permutation 与 内建），避免 SMOTE 对重要性计算产生偏差
"""
import os
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings("ignore")

# -----------------------------
# 配置路径 & 读取数据
# -----------------------------
CSV_PATH = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/ComPare_Vitselect.csv"
RESULT_DIR = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/RF_result/"

os.makedirs(RESULT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# 使用 DataFrame（保留列名）
base_data = df.iloc[:272].reset_index(drop=True)       # 用于训练+CV（217条）
predict_data = df.iloc[272:339].reset_index(drop=True) # 需预测的 55 条

X_base = base_data.drop(columns=["ID", "Class"])   # DataFrame 保留列名
y_base = base_data["Class"].astype(int)

X_predict = predict_data.drop(columns=["ID", "Class"])
predict_ids = predict_data["ID"].values

feature_names = X_base.columns.to_numpy()

# -----------------------------
# KFold 设置
# -----------------------------
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# -----------------------------
# Optuna 目标函数：SMOTE(k_neighbors=1) 防止报错
# -----------------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 25),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    }

    rf = RandomForestClassifier(
        **params,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # SMOTE 设置为 k_neighbors=1（安全、不报错）
    smote = SMOTE(k_neighbors=1, random_state=42)

    # 使用 imblearn 的 Pipeline（SMOTE 只在 fit_resample 阶段生效，不会在 predict 中更改）
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', smote),
        ('rf', rf)
    ])

    cv_scores = []
    # 分折训练与验证（每折都在 train 上做 SMOTE）
    for train_idx, val_idx in skf.split(X_base, y_base):
        X_tr, X_val = X_base.iloc[train_idx], X_base.iloc[val_idx]
        y_tr, y_val = y_base.iloc[train_idx], y_base.iloc[val_idx]

        pipeline.fit(X_tr, y_tr)
        preds = pipeline.predict(X_val)
        cv_scores.append(f1_score(y_val, preds, average="macro"))

    # 返回平均 Macro-F1
    return float(np.mean(cv_scores))


# -----------------------------
# 执行 Optuna 搜索
# -----------------------------
N_TRIALS = 120
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=N_TRIALS)

print("\n🎯 Optuna 搜索结束")
print("Best params:", study.best_params)
print("Best CV Macro-F1:", study.best_value)

best_params = study.best_params

# -----------------------------
# 最终训练：两个模型
# 1) final_pipeline: 包含 SMOTE 的完整 pipeline（用于对 55 条进行预测）
# 2) rf_no_smote_pipeline: 无 SMOTE，仅 scaler + RF（用于计算特征重要性）
# -----------------------------
# 1) 带 SMOTE 的最终 pipeline（用于预测）
final_rf = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
final_smote = SMOTE(k_neighbors=1, random_state=42)
final_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', final_smote),
    ('rf', final_rf)
])

final_pipeline.fit(X_base, y_base)  # 在全部 217 样本上用 SMOTE 训练

# 2) 不含 SMOTE 的 pipeline（用于特征重要性计算）
rf_no_smote = RandomForestClassifier(
    **best_params,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)
rf_no_smote_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', rf_no_smote)
])

rf_no_smote_pipeline.fit(X_base, y_base)  # 在全部 217 样本上训练（无 SMOTE）

# -----------------------------
# 对 55 条样本进行预测并保存
# -----------------------------
X_predict_df = X_predict  # 保持 DataFrame
preds = final_pipeline.predict(X_predict_df)
results_df = pd.DataFrame({
    "ID": predict_ids,
    "Predicted_Class": preds
})
pred_file = os.path.join(RESULT_DIR, "1RF_SMOTE_predictions_218-272.csv")
results_df.to_csv(pred_file, index=False)
print(f"\n✅ 预测结果已保存: {pred_file}")

# -----------------------------
# 特征重要性计算与保存（前150）
# 两种方式：
#  A) RF 内建 feature_importances_
#  B) Permutation Importance（在 rf_no_smote_pipeline 上计算）
# -----------------------------
TOP_N = 150
if TOP_N > len(feature_names):
    TOP_N = len(feature_names)

# A) 内建 feature_importances_
rf_model_trained = rf_no_smote_pipeline.named_steps['rf']
builtin_importances = rf_model_trained.feature_importances_
order_builtin = np.argsort(builtin_importances)[::-1][:TOP_N]
builtin_df = pd.DataFrame({
    "Feature": feature_names[order_builtin],
    "Importance": builtin_importances[order_builtin]
})
builtin_file = os.path.join(RESULT_DIR, f"Top{TOP_N}_RF_builtin_feature_importances.csv")
builtin_df.to_csv(builtin_file, index=False)
print(f"✅ RF 内建 feature_importances_ 已保存: {builtin_file}")

# B) Permutation Importance
print("\n🔁 正在计算 Permutation Importance（可能耗时，取决于特征数量与 repeats）...")
perm_result = permutation_importance(
    rf_no_smote_pipeline,
    X_base,
    y_base,
    scoring="f1_macro",
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)
perm_means = perm_result.importances_mean
order_perm = np.argsort(perm_means)[::-1][:TOP_N]
perm_df = pd.DataFrame({
    "Feature": feature_names[order_perm],
    "Importance": perm_means[order_perm]
})
perm_file = os.path.join(RESULT_DIR, f"Top{TOP_N}_PermutationImportance.csv")
perm_df.to_csv(perm_file, index=False)
print(f"✅ Permutation Importance 前 {TOP_N} 已保存: {perm_file}")

# 为方便对比，也保存完整两个 importance 的 CSV（全部特征）
all_builtin_df = pd.DataFrame({"Feature": feature_names, "Importance": builtin_importances})
all_builtin_df = all_builtin_df.sort_values("Importance", ascending=False).reset_index(drop=True)
all_builtin_file = os.path.join(RESULT_DIR, f"All_RF_builtin_feature_importances.csv")
all_builtin_df.to_csv(all_builtin_file, index=False)

all_perm_df = pd.DataFrame({"Feature": feature_names, "Importance": perm_means})
all_perm_df = all_perm_df.sort_values("Importance", ascending=False).reset_index(drop=True)
all_perm_file = os.path.join(RESULT_DIR, f"All_PermutationImportance.csv")
all_perm_df.to_csv(all_perm_file, index=False)

# -----------------------------
# 保存模型与 scaler
# -----------------------------
joblib.dump(final_pipeline, os.path.join(RESULT_DIR, "RF_SMOTE_pipeline.pkl"))
joblib.dump(rf_no_smote_pipeline, os.path.join(RESULT_DIR, "RF_no_SMOTE_pipeline_for_importance.pkl"))
print("\n✅ 模型已保存：")
print(f"- 带 SMOTE 的预测 pipeline: {os.path.join(RESULT_DIR, 'RF_SMOTE_pipeline.pkl')}")
print(f"- 无 SMOTE 的 importance pipeline: {os.path.join(RESULT_DIR, 'RF_no_SMOTE_pipeline_for_importance.pkl')}")

# -----------------------------
# 小结输出
# -----------------------------
print("\n=== Summary ===")
print(f"- Optuna best params: {study.best_params}")
print(f"- Optuna best CV Macro-F1: {study.best_value:.4f}")
print(f"- Predictions file: {pred_file}")
print(f"- Top {TOP_N} RF built-in importance: {builtin_file}")
print(f"- Top {TOP_N} Permutation Importance: {perm_file}")
print(f"- All built-in importance: {all_builtin_file}")
print(f"- All permutation importance: {all_perm_file}")
print("\nFinished.")
