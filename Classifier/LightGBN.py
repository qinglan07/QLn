import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import os

# ================== 配置 ==================
CSV_PATH = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/ComPare_Vitselect.csv"
SAVE_DIR = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/result/"
os.makedirs(SAVE_DIR, exist_ok=True)

USE_SELECT = True
TOP_K = 120

# ================== 读取数据 ==================
df = pd.read_csv(CSV_PATH)

# 识别 ID 列
id_col = None
for c in df.columns:
    if c.lower() == "id":
        id_col = c
        break

if id_col is None:
    df.insert(0, "ID_for_output", df.index.astype(str))
    id_col = "ID_for_output"

# 前 272 条已标注样本（用于训练/验证）
labeled_df = df.iloc[0:272].reset_index(drop=True)   # 索引 0~271，共 272 行

# 第 274~340 行待预测样本
unlabeled_df = df.iloc[272:340].reset_index(drop=True) # 索引 272~339，共 68 行

if "Class" not in labeled_df.columns:
    raise ValueError("CSV 中缺少 Class 列。")

# ================== 构造特征矩阵 ==================
feature_cols_all = [c for c in labeled_df.columns if c not in [id_col, "Class"]]
X_all = labeled_df[feature_cols_all].values
y_all = labeled_df["Class"].values
X_unlabeled_all = unlabeled_df[feature_cols_all].values

# ================== 80/20 划分（只用训练集做特征选择） ==================
X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# ================== 在训练集上做特征选择（无泄漏） ==================
if USE_SELECT:
    print("=== 使用训练集做特征选择（无泄漏） ===")

    sel_model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=32,
        random_state=42
    )
    sel_model.fit(X_train_raw, y_train)

    importances = sel_model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:TOP_K]

    # 过滤特征列
    selected_feature_cols = [feature_cols_all[i] for i in top_idx]

    X_train = X_train_raw[:, top_idx]
    X_val = X_val_raw[:, top_idx]
    X_all_sel = X_all[:, top_idx]
    X_unlabeled_sel = X_unlabeled_all[:, top_idx]

    print(f"训练集特征选择完成，仅保留 {len(selected_feature_cols)} 个特征")
else:
    print("未使用特征选择")
    selected_feature_cols = feature_cols_all
    X_train, X_val = X_train_raw, X_val_raw
    X_all_sel = X_all
    X_unlabeled_sel = X_unlabeled_all

# ================== Optuna 目标函数 ==================
def objective(trial):
    params = {
        "learning_rate": trial.suggest_float("lr", 0.01, 0.15),
        "num_leaves": trial.suggest_int("num_leaves", 20, 80),
        "feature_fraction": trial.suggest_float("ff", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bf", 0.7, 1.0),
        "bagging_freq": 1,
        "min_child_samples": trial.suggest_int("minleaf", 5, 30),
        "n_estimators": 500,
        "max_depth": -1,
        "verbosity": -1,
        "random_state": 42
    }

    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=5,
        **params
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(50)]
    )

    pred = model.predict(X_val)
    f1 = f1_score(y_val, pred, average="macro")
    return f1

# ================== Optuna 调参 ==================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=40)

best_params = study.best_params
print("Best params:", best_params)

# ================== 用全部 217 条样本训练最终模型 ==================
final_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=5,
    n_estimators=500,
    random_state=42,
    **best_params
)
final_model.fit(X_all_sel, y_all)

# ================== 验证集 Macro-F1 ==================
eval_model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=5,
    n_estimators=500,
    random_state=42,
    **best_params
)
eval_model.fit(X_train, y_train)
pred_val = eval_model.predict(X_val)
val_f1 = f1_score(y_val, pred_val, average="macro")

print("\n===== 验证集 Macro-F1（无泄漏） =====")
print(val_f1)
print(classification_report(
    y_val, pred_val,
    target_names=[f"Class_{i+1}" for i in range(5)]
))

# ================== 预测未标注样本（第 218~272 行） ==================
pred_unlabeled = final_model.predict(X_unlabeled_sel)
pred_unlabeled_output = pred_unlabeled + 1  # 输出成 1~5

out_df = pd.DataFrame({
    "ID": unlabeled_df[id_col].values,
    "Pred_Class": pred_unlabeled_output
})

save_path = os.path.join(SAVE_DIR, "272340_lightGBN.csv")
out_df.to_csv(save_path, index=False)
print("\n预测保存到：", save_path)
