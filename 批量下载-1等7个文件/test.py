import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from skimage.segmentation import random_walker
from tqdm import tqdm
from joblib import Parallel, delayed

# ---------- 1. 读数据 ----------
path = r"C:\Users\31588\Desktop\SJTU\CTP\BreastUSI\250427_YangJiang_43\Capture_L.mha"
img_vol = sitk.GetArrayFromImage(sitk.ReadImage(path))   # (1674,600,650)

# ---------- 2. 工具函数 ----------
def confidence_map(frame, beta):
    """带 beta 参数的 confidence map"""
    I = frame.astype(float)
    I = (I - I.min()) / (I.max() - I.min() + 1e-8)
    markers = np.zeros_like(I, dtype=np.uint8)
    markers[6:10, :] = 1
    markers[-5:, :] = 2
    labels = random_walker(I, markers, beta=beta, tol=5e-2, mode='cg_j')
    return (labels == 1).astype(np.float32)

def delta_one(frame, beta):
    """单帧 δ 值"""
    img = frame.astype(float)
    conf = confidence_map(img, beta)
    brightness = np.clip((img - 10) /
                         (np.percentile(img, 90) - 10 + 1e-8), 0, 1) ** 0.5
    numerator = (conf * brightness).sum()
    denominator = conf.sum() + 1e-8
    return float(numerator / denominator)

# ---------- 3. 主循环 ----------
betas = [5, 10, 15, 20, 30, 60, 100]
plt.figure(figsize=(7,4))
for b in betas:
    print(f'>>> beta = {b}')
    delta_vec = Parallel(n_jobs=20, backend='loky')(
        delayed(delta_one)(f, b) for f in tqdm(img_vol, desc=f'β={b}')
    )
    plt.plot(delta_vec, label=f'beta={b}')

# ---------- 4. 画图 ----------
plt.xlabel('slice index z')
plt.ylabel('δ(z)')
plt.title('δ(z) vs. beta')
plt.legend()
plt.tight_layout()
plt.show()