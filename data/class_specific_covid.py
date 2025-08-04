import os
import torch.utils.data as data
from PIL import Image
import os
import numpy as np
import copy

class ClassSpecificCOVID19(data.Dataset):
    """COVID-19 Chest X-ray Dataset with specific class filter"""
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __init__(self, root, class_idx, split='train', train_ratio=0.8, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.class_idx = class_idx  # 指定类别索引
        self.class_name = self.classes[class_idx]  # 类别名称

        dataset_dir = os.path.join(self.root, 'covid_dataset')
        # 只收集指定类别的数据
        class_data = []
        class_targets = []
        
        cls_dir = os.path.join(dataset_dir, self.class_name)
        if os.path.isdir(cls_dir):
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.png'):
                    class_data.append(os.path.join(cls_dir, img_name))
                    class_targets.append(self.class_idx)
        
        # 将类别数据转为numpy数组并随机打乱
        class_data = np.array(class_data)
        class_targets = np.array(class_targets)
        
        if len(class_data) > 0:
            indices = np.random.permutation(len(class_data))
            class_data = class_data[indices]
            class_targets = class_targets[indices]
            
            # 按比例划分训练集和测试集
            n_train = int(len(class_data) * self.train_ratio)
            
            # 根据split选择相应的数据集
            if self.split == 'train':
                self.data = class_data[:n_train]
                self.targets = class_targets[:n_train]
            else:
                self.data = class_data[n_train:]
                self.targets = class_targets[n_train:]
        else:
            self.data = np.array([])
            self.targets = np.array([])
        
        self.num_class = 4  # 总类别数保持不变
        eps = 0.001

        # 创建软标签
        if len(self.data) > 0:
            if self.split == 'train':
                train_num = len(self.data)
                self.softlabel = np.ones([train_num, self.num_class], dtype=np.float32)*eps/(self.num_class-1)
                for i in range(train_num):
                    self.softlabel[i, self.targets[i]] = 1 - eps
            else:
                test_num = len(self.data)
                self.softlabel = np.ones([test_num, self.num_class], dtype=np.float32)*eps/(self.num_class-1)
                for i in range(test_num):
                    self.softlabel[i, self.targets[i]] = 1 - eps
        else:
            self.softlabel = np.array([])
        
        # 保存原始图片路径和原始索引的映射，用于合并数据集
        self.original_paths = copy.deepcopy(self.data)
        
    def __getitem__(self, index):
        img_path = self.data[index]
        target = int(self.targets[index])
        softlabel = self.softlabel[index]

        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, softlabel, index

    def __len__(self):
        return len(self.data)

    def update_corrupted_label(self, noise_label):
        self.targets[:] = noise_label[:]

    def update_corrupted_softlabel(self, noise_label):
        self.softlabel[:] = noise_label[:]
        
    def get_data_labels(self):
        return self.targets

    def get_data_softlabel(self):
        return self.softlabel
    
    def get_original_paths(self):
        """返回原始图片路径列表，用于数据集合并时的匹配"""
        return self.original_paths


class CombinedCOVID19Dataset(data.Dataset):
    """将四个子类数据集合并为一个完整的数据集"""
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    def __init__(self, class_datasets, transform=None):
        self.transform = transform
        self.class_datasets = class_datasets  # 四个子类数据集列表
        self.num_class = 4
        
        # 合并数据和标签
        all_data = []
        all_targets = []
        all_softlabels = []
        all_original_paths = []
        
        for dataset in class_datasets:
            if len(dataset.data) > 0:
                all_data.extend(dataset.data)
                all_targets.extend(dataset.targets)
                all_softlabels.extend(dataset.softlabel)
                all_original_paths.extend(dataset.get_original_paths())
        
        # 转换为numpy数组
        self.data = np.array(all_data)
        self.targets = np.array(all_targets)
        self.softlabel = np.array(all_softlabels)
        self.original_paths = np.array(all_original_paths)
        
        # 创建路径到索引的映射，用于标签更新
        self.path_to_index = {path: idx for idx, path in enumerate(self.original_paths)}
    
    def __getitem__(self, index):
        img_path = self.data[index]
        target = int(self.targets[index])
        softlabel = self.softlabel[index]

        img = Image.open(img_path).convert('L')
        if self.transform is not None:
            img = self.transform(img)
        return img, target, softlabel, index

    def __len__(self):
        return len(self.data)
    
    def update_from_class_datasets(self):
        """从各个子类数据集更新标签和软标签"""
        for dataset in self.class_datasets:
            if len(dataset.data) > 0:
                for i, path in enumerate(dataset.original_paths):
                    if path in self.path_to_index:
                        idx = self.path_to_index[path]
                        self.targets[idx] = dataset.targets[i]
                        self.softlabel[idx] = dataset.softlabel[i]
    
    def get_data_labels(self):
        return self.targets

    def get_data_softlabel(self):
        return self.softlabel
