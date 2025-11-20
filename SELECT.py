#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SELECT.py（最终优化版）
-------------------------------------------------------------
两阶段特征选择（t-test → Mutual Info） + 随机森林预测
核心逻辑：
1. t-test 阶段：仅使用前217个样本计算，筛选差异显著特征
2. 特征过滤：自动去除方差为0的恒定特征（避免MI误选）
3. MI 阶段：用前217个样本训练，选中高关联特征
4. 模型评估：训练集=前217个样本，验证集=第218-272个样本（无重叠）
5. 结果保存：保存所有样本的选中特征列（无行缺失）
-------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# ------------------- 全局参数（可按需调整） -------------------
VOWELS = ["A", "E", "I", "O", "U", "KA", "PA", "TA"]
TT_DIR = "/home/oem/qinglan7/ICA/DATA/tt/MEL/ViT/"
TEMPLATE_FILE = "A.csv"  # 特征文件模板（替换A为目标发音）
DATA_CSV = "/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/Vit.csv"  # 标签文件路径
OUT_DIR = "/home/oem/qinglan7/ICA/DATA/tt/MEL/ViT/select/"
os.makedirs(OUT_DIR, exist_ok=True)  # 确保输出目录存在

# 特征选择与模型参数
P_VALUE_TH = 0.01  # t-test 显著性阈值
MI_KEEP_LIST = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]  # 待测试的MI特征数
RANDOM_STATE = 0  # 固定随机种子（可复现）
N_ESTIMATORS = 200  # 随机森林树数量
TRAIN_SAMPLE_NUM = 217  # 训练集：前217个样本（1-217）
VAL_SAMPLE_NUM = 55  # 验证集：218-272个样本（共55个）
VAL_START_IDX = TRAIN_SAMPLE_NUM  # 验证集起始索引（217）
VAL_END_IDX = TRAIN_SAMPLE_NUM + VAL_SAMPLE_NUM  # 验证集结束索引（272）
VAR_THRESHOLD = 1e-8  # 方差阈值（小于该值视为恒定特征）

# 结果存储容器
summary = []
all_results = []


# ------------------- 工具函数（增强复用性） -------------------
def filter_constant_features(X, feat_names, var_threshold=VAR_THRESHOLD):
    """过滤恒定特征（方差接近0）"""
    var = X.var(axis=0)  # 计算每列方差
    non_constant_mask = var > var_threshold
    X_filtered = X[:, non_constant_mask]
    feat_names_filtered = feat_names[non_constant_mask]
    removed_feats = feat_names[~non_constant_mask]
    if len(removed_feats) > 0:
        print(f"⚠️ 移除 {len(removed_feats)} 个恒定特征（方差≈0）")
    return X_filtered, feat_names_filtered


# ------------------- 主流程 -------------------
if __name__ == "__main__":
    # 读取所有样本的标签（ID + Class）
    try:
        df_label_all = pd.read_csv(DATA_CSV)
        assert "ID" in df_label_all.columns and "Class" in df_label_all.columns, \
            "DATA.csv 必须包含 'ID' 和 'Class' 列"
        assert len(df_label_all) >= VAL_END_IDX, \
            f"标签文件样本数不足 {VAL_END_IDX} 个（当前仅 {len(df_label_all)} 个）"
    except Exception as e:
        raise ValueError(f"读取标签文件失败：{str(e)}")

    for vowel in VOWELS:
        # 构建当前发音的特征文件路径
        file_path = os.path.join(TT_DIR, TEMPLATE_FILE.replace("A", vowel))
        if not os.path.exists(file_path):
            print(f"⚠️ 文件不存在: {file_path}，跳过该发音")
            continue

        print(f"\n========================= 🔤 处理发音: {vowel} =========================")
        # 读取当前发音的所有样本特征
        try:
            df_feat_all = pd.read_csv(file_path)
            assert "ID" in df_feat_all.columns, f"{file_path} 必须包含 'ID' 列"
        except Exception as e:
            print(f"❌ 读取特征文件失败：{str(e)}，跳过该发音")
            continue

        # ------------------- 关键修复1：分离ID列和数值特征列（明确排除ID） -------------------
        # 数值特征列：所有数值类型列，且排除ID列（避免后续索引操作后找不到列）
        feat_cols_all = df_feat_all.select_dtypes(include=[np.number]).columns.tolist()
        if "ID" in feat_cols_all:
            feat_cols_all.remove("ID")  # 确保数值特征列中不含ID
        feat_cols_all = np.array(feat_cols_all)  # 转成数组方便后续索引
        print(f"原始特征总数：{len(feat_cols_all)}")

        # ------------------- 步骤1：划分固定训练集/验证集（无重叠） -------------------
        # 训练集：前217个样本（1-217）
        df_train_label = df_label_all.iloc[:TRAIN_SAMPLE_NUM].copy()
        train_ids = df_train_label["ID"].values
        print(f"训练集：{len(train_ids)} 个样本（ID：{train_ids[:3]}...{train_ids[-3:]}）")

        # 验证集：第218-272个样本（共55个）
        df_val_label = df_label_all.iloc[VAL_START_IDX:VAL_END_IDX].copy()
        val_ids = df_val_label["ID"].values
        print(f"验证集：{len(val_ids)} 个样本（ID：{val_ids[:3]}...{val_ids[-3:]}）")

        # ------------------- 步骤2：t-test 特征选择（仅用训练集：前217个样本） -------------------
        # 对齐训练集特征（保证ID顺序一致）
        df_feat_train = df_feat_all[df_feat_all["ID"].isin(train_ids)].copy()  # 不提前设索引
        df_feat_train = df_feat_train.set_index("ID").reindex(train_ids).reset_index()  # 先对齐再重置索引

        # ------------------- 关键修复2：提前初始化 valid_train_ids（避免未定义） -------------------
        valid_train_ids = df_feat_train["ID"].values  # 默认使用所有训练集样本ID
        if df_feat_train.isnull().any().any():
            print(f"⚠️ 训练集中存在缺失的特征数据，已自动删除含NaN的行")
            df_feat_train = df_feat_train.dropna()
            # 更新有效训练集ID（仅保留特征非空的样本）
            valid_train_ids = df_feat_train["ID"].values
            # 同步更新训练集标签
            df_train_label = df_train_label[df_train_label["ID"].isin(valid_train_ids)]

        # 提取t-test用的特征矩阵和标签（此时df_feat_train仍有ID列，需排除）
        X_t_train = df_feat_train[feat_cols_all].values
        y_t_train = df_train_label.set_index("ID").loc[valid_train_ids]["Class"].astype(float).values

        # 执行t-test（多类别：每个类别与其他类别对比）
        mask_t = np.zeros(X_t_train.shape[1], dtype=bool)
        unique_cls = np.unique(y_t_train[~np.isnan(y_t_train)])  # 排除NaN标签
        if len(unique_cls) < 2:
            print(f"⚠️ 训练集中仅包含 {len(unique_cls)} 个类别，跳过t-test，使用所有特征")
            mask_t = np.ones(X_t_train.shape[1], dtype=bool)
        else:
            for cls in unique_cls:
                cls_mask = (y_t_train == cls) & (~np.isnan(y_t_train))
                other_mask = (~cls_mask) & (~np.isnan(y_t_train))
                # 执行独立样本t-test（不假设方差相等）
                _, p_vals = ttest_ind(
                    X_t_train[cls_mask], X_t_train[other_mask],
                    axis=0, equal_var=False, nan_policy='omit'
                )
                mask_t |= (p_vals <= P_VALUE_TH)  # 只要一个类别满足就保留特征

        # 应用t-test筛选结果（所有样本都保留筛选后的特征列）
        feat_t = feat_cols_all[mask_t]
        X_t_all = df_feat_all[feat_t].values  # 所有样本的t-test后特征（df_feat_all未改索引，直接选）
        print(f"t-test 筛选后特征数：{len(feat_t)}")

        # ------------------- 步骤3：过滤恒定特征（方差≈0） -------------------
        X_t_filtered, feat_t_filtered = filter_constant_features(X_t_all, feat_t)
        if len(feat_t_filtered) == 0:
            print(f"❌ t-test后无有效特征（全部为恒定特征），跳过该发音")
            continue
        print(f"过滤恒定特征后剩余：{len(feat_t_filtered)} 个特征")

        # ------------------- 步骤4：MI 特征选择（仅用训练集：前217个样本） -------------------
        # 提取训练集的过滤后特征（用于MI计算）
        X_mi_train = df_feat_train[feat_t_filtered].values
        y_mi_train = df_train_label["Class"].astype(int).values

        # ------------------- 步骤5：对齐验证集特征（确保无重叠、无缺失） -------------------
        # 对齐验证集特征（按验证集ID顺序排列）
        df_feat_val = df_feat_all[df_feat_all["ID"].isin(val_ids)].copy()
        df_feat_val = df_feat_val.set_index("ID").reindex(val_ids).reset_index()

        # 提前初始化 valid_val_ids（避免未定义）
        valid_val_ids = df_feat_val["ID"].values
        if df_feat_val.isnull().any().any():
            print(f"⚠️ 验证集中存在缺失的特征数据，已自动删除含NaN的行")
            df_feat_val = df_feat_val.dropna()
            # 更新有效验证集ID
            valid_val_ids = df_feat_val["ID"].values
            # 同步更新验证集标签
            df_val_label = df_val_label[df_val_label["ID"].isin(valid_val_ids)]

        # 提取验证集的过滤后特征（用于模型评估）
        X_val = df_feat_val[feat_t_filtered].values
        y_val = df_val_label.set_index("ID").loc[valid_val_ids]["Class"].astype(int).values
        print(f"有效验证集样本数：{len(y_val)}（原始55个）")

        # 遍历不同的MI特征保留数，评估模型性能
        results = []
        for mi_keep in MI_KEEP_LIST:
            # 计算互信息分数（仅用训练集）
            mi_scores = mutual_info_classif(
                X_mi_train, y_mi_train,
                discrete_features=False,
                n_neighbors=5,  # 减少数值误差
                random_state=RANDOM_STATE
            )
            # 选择MI分数最高的前N个特征（避免超过可用特征数）
            select_num = min(mi_keep, len(mi_scores))
            top_idx = np.argsort(mi_scores)[-select_num:]  # 倒序排序，取Top N
            feat_final = list(feat_t_filtered[top_idx])  # 最终选中的特征名称

            # ------------------- 步骤6：保存所有样本的选中特征 -------------------
            X_final_all = X_t_filtered[:, top_idx]  # 所有样本的最终特征
            df_final = pd.DataFrame(X_final_all, columns=feat_final)
            df_final["ID"] = df_feat_all["ID"].values  # 所有样本的ID
            # 匹配Class标签（处理ID不匹配）
            id_to_class = dict(zip(df_label_all["ID"], df_label_all["Class"]))
            df_final["Class"] = df_final["ID"].map(id_to_class)
            df_final = df_final[["ID", "Class"] + feat_final]

            # 保存文件
            out_file = os.path.join(OUT_DIR, f"{vowel}_{mi_keep}_selectedfeatures_ALL_SAMPLES.csv")
            df_final.to_csv(out_file, index=False, encoding="utf-8")
            print(f"📊 已保存：{os.path.basename(out_file)}（样本数：{len(df_final)}）")

            # ------------------- 步骤7：随机森林训练与评估（训练集→验证集） -------------------
            # 训练集特征（MI筛选后）
            X_train = X_mi_train[:, top_idx]
            # 验证集特征（MI筛选后）
            X_val_final = X_val[:, top_idx]

            # 训练随机森林（仅用训练集）
            model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1  # 并行训练
            )
            model.fit(X_train, y_mi_train)  # 仅用前217个样本训练

            # 验证集预测（仅用第218-272个样本）
            y_pred = model.predict(X_val_final)

            # 计算评估指标
            acc = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average="macro")  # 多类别用macro-F1

            # 记录结果
            print(f"MI_KEEP={mi_keep:<4} → Accuracy={acc:.4f} | Macro-F1={f1:.4f}")
            results.append((mi_keep, acc, f1))
            all_results.append({
                "Vowel": vowel,
                "MI_KEEP": mi_keep,
                "Accuracy": round(acc, 4),
                "Macro_F1": round(f1, 4),
                "Train_Samples": len(X_train),
                "Val_Samples": len(X_val_final)
            })

        # ------------------- 步骤8：记录当前发音的最佳结果 -------------------
        if results:
            best_mi_keep, best_acc, best_f1 = max(results, key=lambda x: x[1])  # 按准确率选最佳
            summary.append((vowel, best_mi_keep, best_acc, best_f1))
            print(f"🏆 {vowel} 最佳结果：MI_KEEP={best_mi_keep} → Acc={best_acc:.4f}, F1={best_f1:.4f}")

        # ------------------- 步骤9：绘制单发音性能折线图 -------------------
        df_plot = pd.DataFrame(results, columns=["MI_KEEP", "Accuracy", "Macro_F1"])
        plt.figure(figsize=(8, 5))
        plt.plot(df_plot["MI_KEEP"], df_plot["Accuracy"], marker="o", linewidth=2, label="Accuracy")
        plt.plot(df_plot["MI_KEEP"], df_plot["Macro_F1"], marker="s", linewidth=2, label="Macro-F1")
        plt.title(f"{vowel}：Performance vs MI Feature Count", fontsize=12)
        plt.xlabel("Number of MI Selected Features", fontsize=10)
        plt.ylabel("Score", fontsize=10)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend(fontsize=10)
        plt.xticks(df_plot["MI_KEEP"], rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{vowel}_performance_curve.png"), dpi=300)
        plt.close()
    # ------------------- 最终汇总 -------------------
    # 保存所有结果到CSV
    df_all_results = pd.DataFrame(all_results).sort_values(["Vowel", "MI_KEEP"])
    df_all_results.to_csv(os.path.join(OUT_DIR, "all_vowels_detailed_results.csv"), index=False, encoding="utf-8")

    # 保存最佳结果汇总
    df_summary = pd.DataFrame(summary, columns=["Vowel", "Best_MI_KEEP", "Best_Accuracy", "Best_Macro_F1"])
    df_summary.to_csv(os.path.join(OUT_DIR, "all_vowels_best_results.csv"), index=False, encoding="utf-8")

    # 打印最佳结果汇总
    print("\n" + "="*50)
    print("所有发音的最佳结果汇总（训练集=1-217，验证集=218-272）")
    print("="*50)
    print(df_summary.to_string(index=False))

    # ------------------- 绘制所有发音的汇总折线图 -------------------
    # 准确率汇总图
    plt.figure(figsize=(12, 6))
    for vowel in VOWELS:
        sub_data = df_all_results[df_all_results["Vowel"] == vowel]
        if not sub_data.empty:
            plt.plot(sub_data["MI_KEEP"], sub_data["Accuracy"], marker="o", linewidth=2, label=vowel)
    plt.title("All Vowels: Accuracy vs MI Feature Count", fontsize=14)
    plt.xlabel("Number of MI Selected Features", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Vowel", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_vowels_accuracy_summary.png"), dpi=300)
    plt.close()

    # Macro-F1汇总图
    plt.figure(figsize=(12, 6))
    for vowel in VOWELS:
        sub_data = df_all_results[df_all_results["Vowel"] == vowel]
        if not sub_data.empty:
            plt.plot(sub_data["MI_KEEP"], sub_data["Macro_F1"], marker="o", linewidth=2, label=vowel)
    #plt.title("All Vowels: Macro-F1 vs MI Feature Count", fontsize=14)
    plt.xlabel("K", fontsize=12)
    plt.ylabel("Avg.F1-score", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(title="Vowel", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "all_vowels_macrof1_summary.png"), dpi=300)
    plt.close()

    print(f"\n✅ 所有发音处理完成！结果已保存至：{OUT_DIR}")