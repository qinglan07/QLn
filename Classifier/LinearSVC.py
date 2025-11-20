import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# -----------------------------
# 1. 路径
# -----------------------------
CSV_PATH = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/ComPare_Vitselect.csv"
SAVE_DIR = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/result/"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# 2. 读取数据
# -----------------------------
df = pd.read_csv(CSV_PATH)

df_labeled = df[df["Class"].notna()].copy()
df_unlabeled = df[df["Class"].isna()].copy()

train_df = df_labeled.iloc[:217]
test_df  = df_labeled.iloc[217:272]

X_train = train_df.drop(columns=["ID", "Class"]).values
y_train = train_df["Class"].astype(int).values

X_test  = test_df.drop(columns=["ID", "Class"]).values
y_test  = test_df["Class"].astype(int).values

# -----------------------------
# 3. 特征标准化（SVM 必须）
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# -----------------------------
# 4. 训练 LinearSVC
# -----------------------------
model = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
model.fit(X_train, y_train)

# -----------------------------
# 5. 测试集 Macro F1
# -----------------------------
y_pred = model.predict(X_test)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("========== LinearSVC 结果 ==========")
print("Test Macro F1:", macro_f1)
print("使用模型参数: C=1.0, class_weight='balanced'")

# -----------------------------
# 6. 预测无标签样本
# -----------------------------
if len(df_unlabeled) > 0:
    X_unlabel = scaler.transform(df_unlabeled.drop(columns=["ID", "Class"]).values)
    pred_unlabel = model.predict(X_unlabel)

    result_df = pd.DataFrame({
        "ID": df_unlabeled["ID"],
        "Class": pred_unlabel
    })

    save_path = os.path.join(SAVE_DIR, "LinearSVC_prediction.csv")
    result_df.to_csv(save_path, index=False)
    print("未标注预测结果已保存:", save_path)
