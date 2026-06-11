import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.segmentation import random_walker
from scipy.ndimage import uniform_filter

# 1. 读图 → 只取第一帧
path = r"C:\Users\31588\Desktop\SJTU\CTP\BreastUSI\250427_BianBaZhuoMa_37\Capture_R.mha"
img_vol = sitk.GetArrayFromImage(sitk.ReadImage(path))      # (1674,600,650)
frame = img_vol[0]                                          # 只用第一帧

# # 2. Shannon 熵（内部归一化到 0-1）
# def shannon_entropy(frame, bins=256):
#     hist, _ = np.histogram(frame, bins=bins, range=(0, 256), density=True)
#     hist += 1e-8
#     return -np.sum(hist * np.log2(hist))

# entropy = shannon_entropy(frame)

# # 3. 对比度（原始灰度级标准差）
# contrast = frame.std()

# 4. SpeckleIdx（原始灰度级，绝对阈值）
def speckle_idx_raw(img, win=9, dark_abs=10):
    mean = ndimage.uniform_filter(img, size=win, mode='reflect')
    std  = np.sqrt(ndimage.uniform_filter(img**2, win) - mean**2)
    mask = mean > dark_abs
    if mask.sum() == 0:
        return 0.0
    return (std[mask] / (mean[mask] + 1e-8)).mean()

speckle = speckle_idx_raw(frame)

# def speckle_idx_crop(frame: np.ndarray, win: int = 9, dark_abs: float = 10.0) -> float:
#     """与 C++ 缩边逻辑完全一致的单帧斑噪指数"""
#     rad = win // 2
#     # 1. 缩边：直接剃掉边缘
#     inner = frame[rad:-rad, rad:-rad]
#     # 2. 滤波（此时窗口永不越界，mode 随意）
#     mean = uniform_filter(inner, size=win, mode='constant')
#     std  = np.sqrt(uniform_filter(inner.astype(np.float64)**2, size=win, mode='constant') - mean**2)
#     # 3. 只统计亮区
#     mask = mean > dark_abs
#     return (std[mask] / (mean[mask] + 1e-8)).mean() if mask.any() else 0.0

# si = speckle_idx_crop(frame, win=9, dark_abs=10)

def speckle_idx_aligned(img, win=9, dark_abs=10.0):
    h, w = img.shape
    rad = win // 2

    # 1. 先算 mean / std，**模式用 constant 0**，后面自己切边界
    mean = ndimage.uniform_filter(img, size=win, mode='constant')
    std  = np.sqrt(ndimage.uniform_filter(img**2, win, mode='constant') - mean**2)

    # 2. 把边界切掉，与 C++ 完全一致
    mean = mean[rad:h-rad, rad:w-rad]
    std  = std[rad:h-rad,  rad:w-rad]
    mask = mean > dark_abs
    if mask.sum() == 0:
        return 0.0
    return (std[mask] / (mean[mask] + 1e-8)).mean()

si = speckle_idx_aligned(frame, win=9, dark_abs=10)

print(si)







# # 5. 置信图 + δ（单帧）
# def confidence_map(frame):
#     I = frame.astype(float)
#     I = (I - I.min()) / (I.max() - I.min() + 1e-8)          # [0,1]
#     markers = np.zeros_like(I, dtype=np.uint8)
#     markers[6:10, :] = 1
#     markers[-5:, :] = 2
#     labels = random_walker(I, markers, beta=25, tol=5e-2, mode='cg_j')
#     return (labels == 1).astype(np.float32)

# def delta_one(img):
#     img = img.astype(float)
#     conf = confidence_map(img)
#     brightness = np.clip((img - 10) / (np.percentile(img, 90) - 10 + 1e-8), 0, 1) ** 0.5
#     numerator   = (conf * brightness).sum()
#     denominator = conf.sum() + 1e-8
#     return float(numerator / denominator)

# delta = delta_one(frame)

# 6. 打印结果
#print(f'Frame 0: δ={delta:.3f}, Entropy={entropy:.4f} bit, Contrast={contrast:.2f}, SpeckleIdx={speckle:.3f}')

# print(f'SpeckleIdx={si:.3f}')