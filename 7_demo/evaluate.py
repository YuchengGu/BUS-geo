import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from skimage.segmentation import random_walker
from tqdm import tqdm
from joblib import Parallel, delayed


# 1. 文件列表
# file_list = [
#     ('20251015185250.txt', '20251015185250'),
#     ('20251015185703.txt', '20251015185703'),
#     ('20251015185905.txt', '20251015185905'),
#     ('20251015190050.txt', '20251015190050'),
#     ('20251015190217.txt', '20251015190217'),
#     ('20251015190642.txt', '20251015190642'),
#     ('20251015190826.txt', '20251015190826'),
#     ('20251015191009.txt', '20251015191009'),
#     ('20251015191154.txt', '20251015191154'),
#     ('20251015191335.txt', '20251015191335'),
#     ('20251015194829.txt', '20251015194829'),
#     ('20251015195047.txt', '20251015195047'),
#     ('20251015195316.txt', '20251015195316'),
#     ('20251015195748.txt', '20251015195748'),
#     ('20251015200109.txt', '20251015200109'),
#     ('20251015200347.txt', '20251015200347'),
#     ('20251015200546.txt', '20251015200546'),
#     ('20251015200806.txt', '20251015200806'),
#     ('20251015201036.txt', '20251015201036'),
#     ('20251015201228.txt', '20251015201228'),
# ]
file_list = [
    ('BianBaZhuoMa_37_L.txt', 'BianBaZhuoMa_37_L'),
    ('BianBaZhuoMa_37_R.txt', 'BianBaZhuoMa_37_R'),
    ('DongBo_46_L.txt', 'DongBo_46_L'),
    ('DongBo_46_R.txt', 'DongBo_46_R'),
    ('QianLongHua_51_L.txt', 'QianLongHua_51_L'),
    ('QianLongHua_51_R.txt', 'QianLongHua_51_R'),
    ('ShenNi_33_L.txt', 'ShenNi_33_L'),
    ('ShenNi_33_R.txt', 'ShenNi_33_R'),
    ('TaoLiNa_38_L.txt', 'TaoLiNa_38_L'),
    ('TaoLiNa_38_R.txt', 'TaoLiNa_38_R'),
    ('YangJiang_43_L.txt', 'YangJiang_43_L'),
    ('YangJiang_43_R.txt', 'YangJiang_43_R'),
    ('ZhangJiaoJiao_29_L.txt', 'ZhangJiaoJiao_29_L'),
    ('ZhangJiaoJiao_29_R.txt', 'ZhangJiaoJiao_29_R'),
    ('ZhouYuan_43_L.txt', 'ZhouYuan_43_L'),
    ('ZhouYuan_43_R.txt', 'ZhouYuan_43_R'),
    #('ZhuMingHui_34_L.txt', 'ZhuMingHui_34_L'),
    #('ZhuMingHui_34_R.txt', 'ZhuMingHui_34_R'),
    ('ZhuShan_32_L.txt', 'ZhuShan_32_L'),
    ('ZhuShan_32_R.txt', 'ZhuShan_32_R'),
    ('ChenJia_29_L.txt','ChenJia_29_L'),
    ('ChenJia_29_R.txt','ChenJia_29_R'),
    ('FanXiaoNing_58_L.txt','FanXiaoNing_58_L'),
    ('FanXiaoNing_58_R.txt','FanXiaoNing_58_R'),
    ('250510_LiuLiJun_33_L.txt', 'LiuLiJun_33_L'),
    ('250510_LiuLiJun_33_R.txt', 'LiuLiJun_33_R'),
    ('250510_ShenMei_45_L.txt', 'ShenMei_45_L'),
    ('250510_ShenMei_45_R.txt', 'ShenMei_45_R'),
    ('250510_SongXueKe_29_L.txt', 'SongXueKe_29_L'),
    ('250510_SongXueKe_29_R.txt', 'SongXueKe_29_R'),
    ('250510_XuYaJun_29_L.txt', 'XuYaJun_29_L'),
    ('250510_XuYaJun_29_R.txt', 'XuYaJun_29_R'),
    ('250510_ZhangQin_59_L.txt', 'ZhangQin_59_L'),
    ('250510_ZhangQin_59_R.txt', 'ZhangQin_59_R'),
    ('250510_ZhangYiFan_26_L.txt', 'ZhangYiFan_26_L'),
    ('250510_ZhangYiFan_26_R.txt', 'ZhangYiFan_26_R'),
    ('250519_ChenHuiFen_29_L.txt', 'ChenHuiFen_29_L'),
    ('250519_ChenHuiFen_29_R.txt', 'ChenHuiFen_29_R'),
    ('250519_DuanChuanPing_43_L.txt', 'DuanChuanPing_43_L'),
    ('250519_DuanChuanPing_43_R.txt', 'DuanChuanPing_43_R'),
    ('250519_FangXia_45_L.txt', 'FangXia_45_L'),
    ('250519_FangXia_45_R.txt', 'FangXia_45_R'),
    ('250519_FanMiFen_60_L.txt', 'FanMiFen_60_L'),
    ('250519_FanMiFen_60_R.txt', 'FanMiFen_60_R'),
    ('250519_GaoJia_42_L.txt', 'GaoJia_42_L'),
    ('250519_GaoJia_42_R.txt', 'GaoJia_42_R'),
    ('250519_JiangHuiFang_63_L.txt', 'JiangHuiFang_63_L'),
    ('250519_JiangHuiFang_63_R.txt', 'JiangHuiFang_63_R'),
    ('250519_LiuXinYang_34_L.txt', 'LiuXinYang_34_L'),
    ('250519_LiuXinYang_34_R.txt', 'LiuXinYang_34_R'),
    ('250519_LiXinLe_25_L.txt', 'LiXinLe_25_L'),
    ('250519_LiXinLe_25_R.txt', 'LiXinLe_25_R'),
    ('250519_SuQian_32_L.txt', 'SuQian_32_L'),
    ('250519_SuQian_32_R.txt', 'SuQian_32_R'),
    ('250519_WangXingXing_36_L.txt', 'WangXingXing_36_L'),
    ('250519_WangXingXing_36_R.txt', 'WangXingXing_36_R'),
    ('250519_YuanSiQi_27_L.txt', 'YuanSiQi_27_L'),
    ('250519_YuanSiQi_27_R.txt', 'YuanSiQi_27_R'),
    ('250519_YuJingJing_39_L.txt', 'YuJingJing_39_L'),
    ('250519_YuJingJing_39_R.txt', 'YuJingJing_39_R'),
    ('250519_YuZhouYi_29_L.txt', 'YuZhouYi_29_L'),
    ('250519_YuZhouYi_29_R.txt', 'YuZhouYi_29_R'),
    ('250519_ZhangQi_45_L.txt', 'ZhangQi_45_L'),
    ('250519_ZhangQi_45_R.txt', 'ZhangQi_45_R'),
    ('250519_ZhangXiaoWei_30_L.txt', 'ZhangXiaoWei_30_L'),
    ('250519_ZhangXiaoWei_30_R.txt', 'ZhangXiaoWei_30_R'),
    ('250519_ZhouQiong_24_L.txt', 'ZhouQiong_24_L'),
    ('250519_ZhouQiong_24_R.txt', 'ZhouQiong_24_R'),
]


# 2. 加载并记录“全局偏移量”
data_chunks = []
label_name_vec = []          # 每条样本对应的文件名字符串
global_offset = []           # 每个文件的第一行在大矩阵里的全局行号
offset = 0
for fname, label_name in file_list:
    if not os.path.isfile(fname):
        raise FileNotFoundError(fname)
    mat = np.loadtxt(fname)
    data_chunks.append(mat)
    label_name_vec.extend([label_name] * mat.shape[0])
    global_offset.append(offset)
    offset += mat.shape[0]

X_raw = np.vstack(data_chunks)
label_name_vec = np.array(label_name_vec)
global_offset = np.array(global_offset)

# 3. 归一化
X = X_raw.copy()
X[:, 0] = (X[:, 0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min())
X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
X[:, 2] = 1 - np.abs(X[:, 2] - X[:, 2].mean()) / (X[:, 2].max() - X[:, 2].min())
X[:, 3] = (X[:, 3].max() - X[:, 3]) / (X[:, 3].max() - X[:, 3].min())

# 4. 熵权 + TOPSIS（同上）
P = X / X.sum(axis=0, keepdims=True)
e = -np.sum(P * np.log(P + 1e-8), axis=0) / np.log(X.shape[0])
d = 1 - e
w = d / d.sum()
W_X = X * w
z_pos = W_X.max(axis=0)
z_neg = W_X.min(axis=0)
d_pos = np.sqrt(((W_X - z_pos)**2).sum(axis=1))
d_neg = np.sqrt(((W_X - z_neg)**2).sum(axis=1))
score = d_neg / (d_pos + d_neg + 1e-8)

# 5. 工具函数：全局索引 -> 文件编号 + 文件内局部行号
def gidx_to_local(gidx):
    """
    返回 (file_idx, local_row_idx)
    file_idx 是 file_list 的下标；local_row_idx 是该文件内第几行（从 0 开始）
    """
    file_idx = np.searchsorted(global_offset, gidx, side='right') - 1
    local_row_idx = gidx - global_offset[file_idx]
    return file_idx, local_row_idx

# 6. 打印结果
print('熵权 w =', w)
print('delta_min=', X_raw[:, 0].min())
print('delta_max=', X_raw[:, 0].max())
print('entropy_min=', X_raw[:, 1].min())
print('entropy_max=', X_raw[:, 1].max())
print('contrast_min=', X_raw[:, 2].min())
print('contrast_mean=', X_raw[:, 2].mean())
print('contrast_max=', X_raw[:, 2].max())
print('speckle_min=', X_raw[:, 3].min())
print('speckle_max=', X_raw[:, 3].max())
print('z_pos=', z_pos)
print('z_neg=', z_neg)