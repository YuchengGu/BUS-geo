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




path = r"C:\Users\31588\Desktop\SJTU\CTP\BreastUSI\250519_DuanChuanPing_43\Capture_L.mha"

# path = r"C:\Users\31588\Desktop\SJTU\CTP\BreastUSI\packet\new_big\2\20251015195748\Capture.mha"

# ① 读图
img_sitk = sitk.ReadImage(path)

# ② 转成 numpy
img_np = sitk.GetArrayFromImage(img_sitk)   # shape == (N, H, W) 或 (N, H, W, C)


print('维度数:', img_np.ndim)
print('总张数 (N):', img_np.shape[0])
print('每张像素尺寸 (H, W):', img_np.shape[1:3])

# ③ 显示第 n 张
n = 808                      # 想看的序号
plt.figure(figsize=(4, 4))
plt.imshow(img_np[n], cmap='gray')
plt.title(f"frame {n}/{img_np.shape[0]-1}")
plt.axis('off')
plt.show()

