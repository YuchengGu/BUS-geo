import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage.segmentation import random_walker
from tqdm import tqdm
from joblib import Parallel, delayed
import cupy as cp
from cupyx.scipy.sparse.linalg import cg




# 1. 读图
path = r"C:\Users\31588\Desktop\SJTU\CTP\BreastUSI\250427_BianBaZhuoMa_37\Capture_R.mha"
img_vol = sitk.GetArrayFromImage(sitk.ReadImage(path))      # (1674,600,650)

# 2.  Shannon 熵（内部归一化到 0-1）
def shannon_entropy(frame, bins=256):
    hist, _ = np.histogram(frame, bins=bins, range=(0, 256), density=True)
    hist += 1e-8
    return -np.sum(hist * np.log2(hist))

entropy_vec = np.array([shannon_entropy(f) for f in img_vol])

# 3. 对比度（原始灰度级标准差）
contrast_vec = np.array([f.std() for f in img_vol])

# 4. SpeckleIdx（原始灰度级，绝对阈值）
def speckle_idx_raw(img, win=9, dark_abs=10):
    mean = ndimage.uniform_filter(img, size=win, mode='reflect')
    std  = np.sqrt(ndimage.uniform_filter(img**2, win) - mean**2)
    mask = mean > dark_abs                 # 只统计信号区
    if mask.sum() == 0:
        return 0.0
    return (std[mask] / (mean[mask] + 1e-8)).mean()

speckle_vec = np.array([speckle_idx_raw(f) for f in img_vol])

# 5.逐帧置信图 + δ（整图平均，可改 ROI）（随机游走）
def confidence_map(frame):
    I = frame.astype(float)
    I = (I - I.min()) / (I.max() - I.min() + 1e-8)          # [0,1]
    h, w = I.shape
    # 标记：顶部 5 行 =1（高置信），底部 5 行 =0（低置信）
    markers = np.zeros_like(I, dtype=np.uint8)
    markers[6:10, :] = 1
    markers[-5:, :] = 2
    labels = random_walker(I, markers, beta=25, tol=5e-2, mode='cg_j')
    return (labels == 1).astype(np.float32)                 # 高置信概率

delta_vec = np.empty(img_vol.shape[0], dtype=np.float32)

def delta_one(frame):
    img = frame.astype(float)
    conf = confidence_map(img)                      # 0 或 1

    # 1. 软阈值：亮区权重 = 归一化亮度
    brightness = np.clip((img - 10) / (np.percentile(img, 90) - 10 + 1e-8), 0, 1) ** 0.5

    # 2. 高置信且亮区的“强度” = conf * bright
    numerator   = (conf * brightness).sum()
    denominator = conf.sum() + 1e-8

    return float(numerator / denominator)

delta_vec = Parallel(n_jobs=20, backend='loky')(
    delayed(delta_one)(f) for f in tqdm(img_vol, desc="Confidence & δ")
)

# 6. 拼成 N×4 矩阵
quality_4d = np.column_stack((delta_vec, entropy_vec, contrast_vec, speckle_vec))

# 7. 终端打印（前 10 帧示意）
np.savetxt('new_big_1_20251015190642.txt', quality_4d, fmt='%.4f %.4f %.2f %.3f',
           header='delta entropy contrast speckle')

for idx, (d, ent, con, spe) in enumerate(quality_4d[:10], 0):
    print(f'Frame {idx:4d}: δ={d:.3f}, Entropy={ent:.4f} bit, Contrast={con:.2f}, SpeckleIdx={spe:.3f}')

