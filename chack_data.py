import pickle
import numpy as np
import matplotlib
print(matplotlib.get_backend())
print(matplotlib.rcParams['backend'])
import matplotlib.pyplot as plt


# ⚠️ 换成你最新的 pkl 文件的绝对路径
FILE_PATH = "/home/ubuntu22/bc_data/gello/0319_153159/2026-03-19T15:32:01.397344.pkl"

def scan_dict(d, indent=0):
    """递归扫描字典里的所有内容并打印结构"""
    for k, v in d.items():
        space = "    " * indent
        if isinstance(v, dict):
            print(f"{space}📁[{k}] (字典, 包含 {len(v)} 个键):")
            scan_dict(v, indent + 1)
        elif hasattr(v, 'shape'):
            print(f"{space}🖼️[{k}] -> Numpy 数组, 形状: {v.shape}, 数据类型: {v.dtype}")
        elif isinstance(v, (list, tuple)):
            print(f"{space}📋 [{k}] -> 列表/元组, 长度: {len(v)}")
        else:
            print(f"{space}📄 [{k}] -> {type(v).__name__}: {v}")

def main():
    try:
        with open(FILE_PATH, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return

    print("="*60)
    print("🔍 1. 数据结构扫描：")
    scan_dict(data)
    print("="*60)

    print("\n⚙️ 2. 核心物理数据：")
    print(f"🦾 【UR5 当前关节】: {np.round(data.get('joint_positions',[]), 3)}")
    print(f"🦾 【UR5 当前关节速度】: {np.round(data.get('joint_velocities',[]), 3)}")
    print(f"🦾 【UR5 末端 rotvec 位姿】: {np.round(data.get('ee_pos_rotvec',[]), 3)}")
    print(f"🦾 【UR5 末端 quat 位姿】: {np.round(data.get('ee_pos_quat',[]), 3)}")
    print(f"🎯 【发送到 UR5 的关节指令】: {np.round(data.get('control',[]), 3)}")
    print(f"🏋️ 【六维力读数】: {np.round(data.get('force',[]), 3)}")
    if 'gripper_position' in data:
        print(f"⚠️ 【夹爪字段】: {np.round(data.get('gripper_position',[]), 3)}")

    print("\n📸 3. 正在生成多模态仪表盘...")
    
    # 收集要显示的图片
    images =[]
    
    if 'D405_rgb' in data:
        # Matplotlib 默认就是 RGB 格式，所以直接塞进去，颜色绝对正确！
        images.append(("Camera 1: D405 RGB", data['D405_rgb'], None))
        
    if 'D405_depth' in data:
        # 把 (H, W, 1) 的深度图降维成 (H, W)，Matplotlib 会自动帮我们涂上热力图颜色！
        depth_2d = np.squeeze(data['D405_depth']) 
        images.append(("Camera 1: D405 Depth", depth_2d, 'jet'))
        
    if 'Orbbec_rgb' in data:
        images.append(("Camera 2: Orbbec RGB", data['Orbbec_rgb'], None))
        
    if 'Orbbec_depth' in data:
        depth_2d = np.squeeze(data['Orbbec_depth'])
        images.append(("Camera 2: Orbbec Depth", depth_2d, 'jet'))
        
    if 'Ultrasound_rgb' in data:
        # 强制转灰度 + 强制指定 gray 色图
        us_img = data['Ultrasound_rgb']
        us_gray = us_img[..., 0]  # 取单通道

        #us_gray  = cv2.flip(us_gray  , 0)

        #us_gray  = us_gray [0:1080, 0:1920]

        # 这里特殊处理：直接当成深度图那样的分支，强制 cmap
        images.append(("Ultrasound", us_gray, 'gray'))

    num_imgs = len(images)
    if num_imgs == 0:
        print("❌ 没有找到任何图像数据！")
        return

    # 创建一个大画板，把所有图片拼在一起
    # 根据图片数量自动调整画板比例 (一行 N 列)
    fig, axes = plt.subplots(1, num_imgs, figsize=(5 * num_imgs, 5))
    
    # 如果只有一张图，axes 不是列表，这里做个兼容
    if num_imgs == 1:
        axes = [axes]
        
    # 循环把图片画上去
    for ax, (title, img, cmap) in zip(axes, images):
        if cmap is not None:  # 如果检测到是深度图
            # 1. 过滤掉无效值 (0) 和 极其离谱的噪点 (>10000mm)
            valid_pixels = img[(img > 0) & (img < 10000)]
            
            if len(valid_pixels) > 0:
                # 2. 掐头去尾，取 2% 和 98% 分位数作为颜色的上下限
                vmin = np.percentile(valid_pixels, 2)
                vmax = np.percentile(valid_pixels, 98)
            else:
                vmin, vmax = 0, 1000 # 防崩溃保底值
                
            # 3. 按照我们计算好的真实有效区间来涂色！
            ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        else:
            # 彩色图正常显示
            ax.imshow(img)
            
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off') # 关掉坐标轴刻度

    plt.tight_layout() # 自动调整间距
    print("\n👉 提示：看完后，直接关闭弹出的画板窗口，程序就会自动结束！")
    
    # 显示画板！(绝对安全，不会死锁)
    plt.show()
    print("✅ 完美退出！")

if __name__ == "__main__":
    main()
