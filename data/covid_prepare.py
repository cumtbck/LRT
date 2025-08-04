import os
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

class COVID19(data.Dataset):
    """COVID-19 Chest X-ray Dataset"""
    classes = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __init__(self, root, split='train', train_ratio=0.8, transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio


        dataset_dir = os.path.join(self.root, 'covid_dataset')
        # 按类别收集数据
        class_data = {cls: [] for cls in self.classes}
        class_targets = {cls: [] for cls in self.classes}
        
        for cls in self.classes:
            cls_dir = os.path.join(dataset_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.png'):
                    class_data[cls].append(os.path.join(cls_dir, img_name))
                    class_targets[cls].append(self.class_to_idx[cls])
        
        # 分层抽样
        self.data = []
        self.targets = []
        train_data = []
        train_targets = []
        test_data = []
        test_targets = []
        
        for cls in self.classes:
            if not class_data[cls]:  # 跳过没有数据的类别
                continue
                
            # 将类别数据转为numpy数组并随机打乱
            cls_data = np.array(class_data[cls])
            cls_targets = np.array(class_targets[cls])
            indices = np.random.permutation(len(cls_data))
            cls_data = cls_data[indices]
            cls_targets = cls_targets[indices]
            
            # 按比例划分训练集和测试集
            n_train = int(len(cls_data) * self.train_ratio)
            train_data.extend(cls_data[:n_train])
            train_targets.extend(cls_targets[:n_train])
            test_data.extend(cls_data[n_train:])
            test_targets.extend(cls_targets[n_train:])
        
        # 根据split选择相应的数据集
        if self.split == 'train':
            self.data = np.array(train_data)
            self.targets = np.array(train_targets)
        else:
            self.data = np.array(test_data)
            self.targets = np.array(test_targets)
        
        self.num_class = len(self.classes)
        eps = 0.001

        # 创建软标签
        if self.split == 'train':
            train_num = len(self.data)
            self.softlabel = np.ones([train_num, self.num_class], dtype=np.float32)*eps/self.num_class
            for i in range(train_num):
                self.softlabel[i, self.targets[i]] = 1 - eps
        else:
            test_num = len(self.data)
            self.softlabel = np.ones([test_num, self.num_class], dtype=np.float32)*eps/self.num_class
            for i in range(test_num):
                self.softlabel[i, self.targets[i]] = 1 - eps
        
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
        
    def modify_selected_data(self, modified_data, indices):
        self.data[indices] = modified_data

    def modify_selected_label(self, modified_label, indices):
        temp = np.array(self.targets)
        temp[indices] = modified_label
        self.targets = list(temp)

    def modify_selected_softlabel(self, modified_softlabel, indices):
        self.softlabel[indices] = modified_softlabel

    def update_selected_data(self, selected_indices):
        self.data = self.data[selected_indices]

        self.targets = np.array(self.targets)
        self.targets = self.targets[selected_indices]
        self.targets = self.targets.tolist()

    def ignore_noise_data(self, noisy_data_indices):
        total = len(self.data)
        remain = list(set(range(total)) - set(noisy_data_indices))
        remain = np.array(remain)

        self.data = self.data[remain]
        self.targets = np.array(self.targets)
        self.targets = self.targets[remain]
        self.targets = self.targets.tolist()
        self.softlabel = self.softlabel[remain]

    def get_data_labels(self):
        return self.targets

    def get_data_softlabel(self):
        return self.softlabel
