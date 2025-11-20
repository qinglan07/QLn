import glob
import warnings
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import opensmile
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------- 1. 基础路径 ----------------------
BASE_DIR = Path("/home/oem/qinglan7/ICA/DATA/tt/")
SUBSETS = ["A", "E", "I", "O", "U", "KA", "PA", "TA"]

DATA_ORDER_CSV = Path("/home/oem/qinglan7/ICA/DATA/tt/DATA.CSV/DATA.csv")  # ID 顺序文件
OUT_DIR = Path("/home/oem/qinglan7/ICA/DATA/tt/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------- 2. 读取 DATA.csv（用于排序） ----------------------
print("🛈 读取 ID 顺序文件 (DATA.csv) …")
data_df = pd.read_csv(DATA_ORDER_CSV)
if "ID" not in data_df.columns:
    raise ValueError("❌ DATA.csv 中必须包含 'ID' 列")

# 去掉前缀 "ID"，统一为整数形式字符串
DATA_ORDER_INT = [str(int(i.replace("ID", ""))) for i in data_df["ID"].tolist()]
print(f"DATA.csv 中共 {len(DATA_ORDER_INT)} 条 ID（将用于排序）")

# ---------------------- 3. 初始化 opensmile ----------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

# ============================================================
#           ⭐ 主循环：逐个子文件夹，提取特征 + 排序 ⭐
# ============================================================
for subset in SUBSETS:
    print(f"\n============================")
    print(f"▶▶ 处理子文件夹: {subset}")
    print(f"============================")

    AUDIO_DIR = BASE_DIR / subset
    wav_files = sorted(glob.glob(str(AUDIO_DIR / "*.wav")),
                       key=lambda x: int(Path(x).stem.replace("ID", "").split("_")[0]))

    print(f"在 {subset} 中找到 {len(wav_files)} 个音频")

    features_list = []
    ids_list = []

    # ----------- 提取特征 -----------
    for wav_path in tqdm(wav_files, desc=f"{subset} 提取中"):
        fname = Path(wav_path).stem   # e.g., ID004_xxx
        try:
            file_id = str(int(fname.replace("ID", "").split("_")[0]))
        except:
            print(f"⚠ 无法解析ID: {fname}")
            continue

        try:
            features = smile.process_file(wav_path).values.flatten()
        except Exception as e:
            print(f"❌ 特征提取失败: {wav_path} → {e}")
            continue

        features_list.append(features)
        ids_list.append(file_id)

    # ----------- 构建 DataFrame -----------
    if len(features_list) == 0:
        print(f"❌ {subset} 没有成功提取到任何特征，跳过")
        continue

    X = pd.DataFrame(features_list, index=ids_list, columns=smile.feature_names)

    # ----------- 按 DATA.csv 的 ID 顺序排序 -----------
    X_sorted = X.reindex(DATA_ORDER_INT).dropna()
    print(f"📌 排序后剩余 {X_sorted.shape[0]} 条（顺序完全与 DATA.csv 一致）")

    # ----------- 保存原始特征 -----------
    raw_df = pd.concat([
        pd.Series(X_sorted.index, name="ID"),
        X_sorted.reset_index(drop=True)
    ], axis=1)

    raw_path = OUT_DIR / f"{subset}_ComParE_raw.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"📌 原始特征保存至 {raw_path}")

    # ----------- 标准化 -----------
    scaler = StandardScaler()
    X_std = pd.DataFrame(scaler.fit_transform(X_sorted), columns=X_sorted.columns, index=X_sorted.index)

    std_df = pd.concat([
        pd.Series(X_std.index, name="ID"),
        X_std.reset_index(drop=True)
    ], axis=1)

    std_path = OUT_DIR / f"{subset}_ComParE_std.csv"
    std_df.to_csv(std_path, index=False)
    print(f"📌 标准化特征保存至 {std_path}")

print("\n🎉 全部分组特征提取完成（顺序已完全对齐 DATA.csv）！")
