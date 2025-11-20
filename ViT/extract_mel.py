import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ------------------- 路径设置 -------------------
ROOT_DIR = "/home/oem/qinglan7/ICA/DATA/tt/"  # 原始音频路径
OUT_DIR = "/home/oem/qinglan7/ICA/DATA/tt/MEL/"  # 彩色 Mel 输出路径
os.makedirs(OUT_DIR, exist_ok=True)

# 发音列表
PHONEMES = ["A", "E", "I", "O", "U", "KA", "PA", "TA"]


# ------------------- 彩色 Mel 图生成函数 -------------------
def audio_to_mel_image_color(audio_path, img_size=224):
    """
    读取音频 -> 生成彩色 Mel 频谱图 -> 转 224x224 RGB 图
    """
    # 1. 读取音频
    y, sr = librosa.load(audio_path, sr=16000)

    # 2. 生成 Mel 频谱
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=1024, hop_length=256, n_mels=img_size)
    S_db = librosa.power_to_db(S, ref=np.max)

    # 3. 使用 matplotlib 生成彩色图
    fig = plt.figure(figsize=(4, 4), dpi=img_size // 4)
    plt.axis('off')
    librosa.display.specshow(S_db, sr=sr, hop_length=256, cmap='magma')
    fig.canvas.draw()

    # 4. 转 numpy
    img_arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_arr = img_arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    # 5. 调整大小到 224x224
    img = Image.fromarray(img_arr)
    img = img.resize((img_size, img_size))

    return img


# ------------------- 批量处理 -------------------
for ph in PHONEMES:
    in_folder = os.path.join(ROOT_DIR, ph)
    out_folder = os.path.join(OUT_DIR, ph)
    os.makedirs(out_folder, exist_ok=True)

    print(f"🔵 处理发音：{ph}")

    for file in os.listdir(in_folder):
        if not file.lower().endswith(".wav"):
            continue

        audio_path = os.path.join(in_folder, file)
        mel_img = audio_to_mel_image_color(audio_path, img_size=224)

        out_name = os.path.splitext(file)[0] + ".png"
        mel_img.save(os.path.join(out_folder, out_name))

print("✅ 所有发音彩色 Mel 图生成完成！")
