import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from collections import Counter
import os

# ================= 配置 =================
CSV_PATH = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/ComPare_Vitselect.csv"
SAVE_DIR = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/result/"
os.makedirs(SAVE_DIR, exist_ok=True)

TOP_K = 140  # 特征选择保留 top-k 特征

# ================= 读取数据 =================
df = pd.read_csv(CSV_PATH)

# ID 列处理
id_col = None
for c in df.columns:
    if c.lower() == "id":
        id_col = c
        break
if id_col is None:
    df.insert(0, "ID_for_output", df.index.astype(str))
    id_col = "ID_for_output"

# ================= 切分训练/预测 =================
train_df = df.iloc[1:274].reset_index(drop=True)      # 第2行到第274行，训练/验证样本
unlabeled_df = df.iloc[274:341].reset_index(drop=True) # 第275行到第341行，待预测样本

# ================= 构建 X/y =================
feature_cols = [c for c in train_df.columns if c not in [id_col, "Class"]]
X_all = train_df[feature_cols].values
y_all = train_df["Class"].astype(int).values  # 保持 0~4

X_unlabeled_all = unlabeled_df[feature_cols].values

# ================== 80/20 划分 ==================
X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
)

# ================== 在训练集上做特征选择 ==================
sel_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.03,
    depth=6,
    verbose=0
)
sel_model.fit(X_train, y_train)
importances = sel_model.get_feature_importance()
top_idx = np.argsort(importances)[::-1][:TOP_K]

# 只保留 top-k 特征
feature_cols = [feature_cols[i] for i in top_idx]
X_train = X_train[:, top_idx]
X_val = X_val[:, top_idx]
X_all = X_all[:, top_idx]
X_unlabeled_all = X_unlabeled_all[:, top_idx]

print(f"特征选择完成，保留 top {len(feature_cols)} 个特征。")

# ================== 处理类别不平衡 ==================
counter = Counter(y_train)
max_count = max(counter.values())
class_weights = [max_count / counter.get(i, 1) for i in range(5)]
print("Class weights:", {i: class_weights[i] for i in range(5)})

# ================== 训练 CatBoost ==================
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    early_stopping_rounds=50,
    verbose=50,
    random_seed=42
)
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# ================== 验证集评估 ==================
pred_val = model.predict(X_val)
val_f1 = f1_score(y_val, pred_val, average="macro")
print("验证集 Macro-F1 =", val_f1)

report = classification_report(
    y_val, pred_val,
    target_names=[f"Class_{i+1}" for i in range(5)],
    digits=4
)
print("\n===== 每类别 F1 =====")
print(report)

# ================== 在全部训练集上训练最终模型 ==================
final_pool = Pool(X_all, y_all)
final_model = CatBoostClassifier(
    iterations=model.get_best_iteration(),
    learning_rate=0.05,
    depth=6,
    loss_function='MultiClass',
    class_weights=class_weights,
    verbose=0,
    random_seed=42
)
final_model.fit(final_pool)

# ================== 预测未标注样本 ==================
pred_unlabeled = final_model.predict(X_unlabeled_all)
pred_unlabeled_output = pred_unlabeled.flatten() + 1  # 输出 1~5

out_df = pd.DataFrame({
    "ID": unlabeled_df[id_col].values,
    "Pred_Class": pred_unlabeled_output
})
save_path = os.path.join(SAVE_DIR, "274_340_CatBoost_final.csv")
out_df.to_csv(save_path, index=False)
print("预测已保存到：", save_path)
