"""
批量提取 BreastUSI 全部病例 L/R 的 4 个质量指标：
delta | entropy | contrast | speckle
结果命名为 <case>_<L/R>.txt 并保存在 ROOT 目录下。
"""
import os
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.segmentation import random_walker
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------- 0. 根目录 ----------
ROOT = r"C:\Users\31588\Desktop\SJTU\CTP\BreastUSI"

# ---------- 1. 病例列表 ----------
# CASES = [
#     "250510_LiuLiJun_33", "250510_ShenMei_45",
#     "250510_SongXueKe_29", "250510_XuYaJun_29",
#     "250510_ZhangQin_59", "250510_ZhangYiFan_26", "250519_ChenHuiFen_29",
#     "250519_DuanChuanPing_43", "250519_FangXia_45", "250519_FanMiFen_60",
#     "250519_GaoJia_42", "250519_JiangHuiFang_63", "250519_LiuXinYang_34",
#     "250519_LiXinLe_25", "250519_SuQian_32", "250519_WangXingXing_36",
#     "250519_YuanSiQi_27", "250519_YuJingJing_39", "250519_YuZhouYi_29",
#     "250519_ZhangQi_45", "250519_ZhangXiaoWei_30", "250519_ZhouQiong_24",
# ]
# #LiNan, ZhaiWenQin

CASES = [
    # new_big
    (r"new_big\1", "20251015190642"), (r"new_big\1", "20251015190826"),
    (r"new_big\1", "20251015191009"), (r"new_big\1", "20251015191154"),
    (r"new_big\1", "20251015191335"), (r"new_big\2", "20251015194829"),
    (r"new_big\2", "20251015195047"), (r"new_big\2", "20251015195316"),
    (r"new_big\2", "20251015195748"), (r"new_big\2", "20251015200109"),
    # new_small
    (r"new_small\1", "20251015185250"), (r"new_small\1", "20251015185703"),
    (r"new_small\1", "20251015185905"), (r"new_small\1", "20251015190050"),
    (r"new_small\1", "20251015190217"), (r"new_small\2", "20251015200347"),
    (r"new_small\2", "20251015200546"), (r"new_small\2", "20251015200806"),
    (r"new_small\2", "20251015201036"), (r"new_small\2", "20251015201228"),
]

# ---------- 2. 单文件跑 4 个指标 ----------
def process_one(mha_path):
    img_vol = sitk.GetArrayFromImage(sitk.ReadImage(mha_path))  # (Z,H,W)

    # 2.1 Shannon 熵
    def entropy(f, bins=256):
        h, _ = np.histogram(f, bins=bins, range=(0, 256), density=True)
        h += 1e-8
        return -np.sum(h * np.log2(h))
    ent_vec = np.array([entropy(f) for f in img_vol])

    # 2.2 对比度
    con_vec = np.array([f.std() for f in img_vol])

    # 2.3 Speckle index
    def speckle(f, win=9, dark=10):
        mu = ndimage.uniform_filter(f, size=win, mode='reflect')
        std = np.sqrt(ndimage.uniform_filter(f**2, win) - mu**2)
        mask = mu > dark
        return (std[mask]/(mu[mask]+1e-8)).mean() if mask.any() else 0.0
    spe_vec = np.array([speckle(f) for f in img_vol])

    # 2.4 confidence-map delta
    def conf_map(f):
        I = (f - f.min()) / (f.max() - f.min() + 1e-8)
        m = np.zeros_like(I, dtype=np.uint8)
        m[6:10, :] = 1
        m[-5:, :] = 2
        lab = random_walker(I, m, beta=25, tol=5e-2, mode='cg_j')
        return (lab == 1).astype(np.float32)

    def delta_one(f):
        img = f.astype(float)
        conf = conf_map(img)
        bright = np.clip((img - 10) /
                         (np.percentile(img, 90) - 10 + 1e-8), 0, 1)**0.5
        return (conf * bright).sum() / (conf.sum() + 1e-8)
    delta_vec = np.array([delta_one(f) for f in img_vol])

    return np.column_stack((delta_vec, ent_vec, con_vec, spe_vec))

# ---------- 3. 拼出所有 L/R 路径 ----------
# tasks = []
# for case in CASES:
#     for lr in ('L', 'R'):
#         mha = os.path.join(ROOT, case, f"Capture_{lr}.mha")
#         if os.path.isfile(mha):
#             tasks.append(mha)
#         else:
#             print('[warn]', mha, 'not found, skip')


tasks = []
for sub, t in CASES:
    mha = os.path.join(ROOT, "packet", sub, t, "Capture.mha")
    if os.path.isfile(mha):
        tasks.append(mha)
    else:
        print('[warn]', mha, 'not found, skip')


# ---------- 4. 并行跑（输出到 BreastUSI 根目录） ----------
# def safe_proc(path):
#     try:
#         qual = process_one(path)
#         case_name = os.path.basename(os.path.dirname(path))   # 250510_LiNan_33
#         lr_flag = os.path.basename(path).split('.')[0][-1]    # L 或 R
#         out_name = f"{case_name}_{lr_flag}.txt"
#         out_txt = os.path.join(ROOT, out_name)
#         np.savetxt(out_txt, qual, fmt='%.4f %.4f %.2f %.3f',
#                    header='delta entropy contrast speckle')
#         print('[done]', out_name)
#     except Exception as e:
#         print('[error]', path, e)

# # 总进度条
# Parallel(n_jobs=16, backend='loky')(
#     delayed(safe_proc)(p) for p in tqdm(tasks, desc='Total cases L/R')
# )


def safe_proc(path):
    try:
        qual = process_one(path)              # 算法部分完全不动
        folder_name = os.path.basename(os.path.dirname(path))  # 时间戳
        out_name = f"{folder_name}.txt"
        out_txt = os.path.join(ROOT, out_name)
        np.savetxt(out_txt, qual, fmt='%.4f %.4f %.2f %.3f',
                   header='delta entropy contrast speckle')
        print('[done]', out_name)
    except Exception as e:
        print('[error]', path, e)

Parallel(n_jobs=16, backend='loky')(
    delayed(safe_proc)(p) for p in tqdm(tasks, desc='Total')
)
