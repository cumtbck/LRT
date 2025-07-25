import os
import torch.utils.data as data
from PIL import Image
import os
import numpy as np

class COVID19(data.Dataset):
    """COVID-19 Chest X-ray Dataset"""
    classes = ['Covid', 'Lung_opacity', 'Normal', 'Viral_pneumonia']
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __init__(self, root, split='train', train_ratio=0.8, trust_ratio=0.1,transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.split = split
        self.train_ratio = train_ratio
        self.trust_ratio = trust_ratio

        # 适配 data/covid_dataset 目录结构
        dataset_dir = os.path.join(self.root, 'covid_dataset')
        self.data = []
        self.targets = []
        for cls in self.classes:
            cls_dir = os.path.join(dataset_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.endswith('.png'):
                    self.data.append(os.path.join(cls_dir, img_name))
                    self.targets.append(self.class_to_idx[cls])

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.num_class = len(self.classes)

        if self.split != 'test':
            num_data=len(self.data)
            trust_num = int(num_data * self.trust_ratio)
            remain = num_data - trust_num
            train_num = int(remain * self.train_ratio)
            if self.split == 'train':
                self.data = self.data[:train_num]
                self.targets = self.targets[:train_num]
                # Add softlabel here
                self.softlabel = np.ones([train_num, self.num_class], dtype=np.int32)
                for i in range(train_num):
                    self.softlabel[i, self.targets[i]] = 1
            elif self.split == 'val':
                self.data = self.data[train_num:remain]
                self.targets = self.targets[train_num:remain]
                self.softlabel = np.ones([(remain-train_num), self.num_class], dtype=np.int32)
                for i in range(remain - train_num):
                    self.softlabel[i, self.targets[i]] = 1
            else:
                self.data = self.data[remain:]
                self.targets = self.targets[remain:]
                self.softlabel = np.ones([(num_data-remain), self.num_class], dtype=np.int32)
                for i in range(num_data-remain):
                    self.softlabel[i, self.targets[i]] = 1 
        else:
            num_data = len(self.data)
            self.softlabel = np.ones([num_data, self.num_class], dtype=np.int32)
            for i in range(num_data):
                self.softlabel[i, self.targets[i]] = 1 


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
