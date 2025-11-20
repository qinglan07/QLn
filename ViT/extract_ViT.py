import os
import torch
import timm
import pandas as pd
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# -----------------------
# 路径设置
# -----------------------
MEL_ROOT = "/home/oem/qinglan7/ICA/DATA/tt/MEL"            # 你保存 mel 图像的路径
OUT_DIR  = "/home/oem/qinglan7/ICA/DATA/tt/MEL/ViT"         # 输出 CSV
MODEL_PATH = "/home/oem/qinglan7/audio_label/icassp2/model/pytorch_model.bin"

os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------
# 加载离线 ViT 模型
# -----------------------
print("🔁 Loading offline ViT model ...")

model = timm.create_model(
    "vit_base_patch16_224.augreg2_in21k_ft_in1k",
    pretrained=False
)

state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

print("✅ ViT model loaded!")

# -----------------------
# 图像预处理
# -----------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    )
])

# -----------------------
# 需要遍历的发音文件夹
# -----------------------
PHONEMES = ["A", "E", "I", "O", "U", "KA", "PA", "TA"]


# =======================
#    提取单张图像特征
# =======================
def extract_feature(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model.forward_features(x)   # [1, 197, 768]

    cls_feat = feat[:, 0, :].squeeze().numpy()  # 取 CLS token，768 维
    return cls_feat


# =======================
#   主循环：遍历所有文件夹
# =======================
for ph in PHONEMES:
    print(f"\n================= 🔤 Processing: {ph} =================")

    folder = os.path.join(MEL_ROOT, ph)
    out_csv = os.path.join(OUT_DIR, f"{ph}.csv")

    rows = []
    ids = []

    files = sorted(os.listdir(folder))

    for fname in tqdm(files):
        if not (fname.endswith(".png") or fname.endswith(".jpg")):
            continue

        img_path = os.path.join(folder, fname)

        # 只保留 ID 的数字部分，例如 "ID004.png" → "004"
        base = os.path.splitext(fname)[0]
        id_num = "".join([c for c in base if c.isdigit()])

        feat = extract_feature(img_path)

        ids.append(id_num)
        rows.append(feat)

    # 保存 CSV — 第一列是 ID，后面是 768 维特征
    df = pd.DataFrame(rows)
    df.insert(0, "ID", ids)
    df.to_csv(out_csv, index=False)

    print(f"✅ Saved: {out_csv}")

print("\n🎉 All phoneme features extracted successfully!")
