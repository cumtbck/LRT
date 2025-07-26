import os
import numpy as np
from PIL import Image
from tqdm import tqdm

def calculate_mean_std():
    # 定义数据集根目录
    dataset_dir = './covid_dataset'
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    
    # 初始化变量
    pixel_sum = 0
    pixel_square_sum = 0
    num_pixels = 0
    
    # 遍历所有图片
    print("正在计算均值和标准差...")
    for cls in classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.isdir(cls_dir):
            print(f"警告: {cls_dir} 不存在")
            continue
            
        # 使用tqdm显示进度条
        for img_name in tqdm(os.listdir(cls_dir), desc=f"处理 {cls} 类别"):
            if img_name.endswith('.png'):
                img_path = os.path.join(cls_dir, img_name)
                
                # 读取图片并转换为灰度图
                with Image.open(img_path) as img:
                    img_gray = img.convert('L')
                    img_array = np.array(img_gray) / 255.0  # 归一化到[0,1]
                    
                    # 更新统计值
                    pixel_sum += img_array.sum()
                    pixel_square_sum += (img_array ** 2).sum()
                    num_pixels += img_array.size
    
    # 计算均值
    mean = pixel_sum / num_pixels
    
    # 计算标准差
    std = np.sqrt(pixel_square_sum / num_pixels - mean ** 2)
    
    print(f"\n数据集统计信息:")
    print(f"总像素数: {num_pixels}")
    print(f"均值: {mean:.4f}")
    print(f"标准差: {std:.4f}")
    
    return mean, std

if __name__ == '__main__':
    mean, std = calculate_mean_std()
