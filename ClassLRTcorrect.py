import os
import sys
import torch
import torch.nn as nn
import torch.nn.parallel
import random
import argparse
import numpy as np
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from tqdm import tqdm
import datetime
import math
from termcolor import cprint

# 导入自定义模块
from data.class_specific_covid import ClassSpecificCOVID19, CombinedCOVID19Dataset
from Utils.noise import noisify_with_P, noisify_pairflip, noisify_covid_asymmetric
from network.net import weight_init, CNN9LAYER, Net
from network.resnet import resnet18, resnet34
from network.preact_resnet import preact_resnet18, preact_resnet34, preact_resnet101, initialize_weights, conv_init
from network.toynet import ToyNet
from network.robust_net import RobustNet, MixupRobustNet, EnsembleRobustNet
from data.utils import ANL_CE_Loss
from data.loss import (
    MAE_Loss, MSE_Loss, CE_Loss, RCE_Loss, NCE_Loss, NNCE_Loss,
    SCE_Loss, GCE_Loss, FL_Loss, NFL_Loss, NNFL_Loss,
    AGCE_Loss, AUE_Loss, AExp_Loss,
    NCE_MAE_Loss, NCE_RCE_Loss, NFL_MAE_Loss, NFL_RCE_Loss,
    ANL_NCE_Loss, ANL_FL_Loss
)


def check_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir


def _init_fn(worker_id):
    np.random.seed(77 + worker_id)


def updateA(s, h, rho=0.9):
    '''
    Used to calculate retroactive loss

    Input
    s : output after softlayer of NN for a specific former epoch
    h : logrithm of output after softlayer of NN at current epoch

    Output
    result : retroactive loss L_retro
    A : correction matrix
    '''
    eps = 1e-4
    h = torch.tensor(h, dtype=torch.float32).reshape(-1, 1)
    s = torch.tensor(s, dtype=torch.float32).reshape(-1, 1)
    A = torch.ones(len(s), len(s))*eps
    A[s.argmax(0)] = rho - eps/(len(s)-1)
    result = -((A.matmul(s)).t()).matmul(h)

    return result, A


def lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta):
    '''
    The LRT correction scheme.
    pred_softlabels_bar is the prediction of the network which is compared with noisy label y_tilde.
    If the LR is smaller than the given threshhold delta, we reject LRT and flip y_tilde to prediction of pred_softlabels_bar

    Input
    pred_softlabels_bar: rolling average of output after softlayers for past 10 epochs. Could use other rolling windows.
    y_tilde: noisy labels at current epoch
    delta: LRT threshholding

    Output
    y_tilde : new noisy labels after cleanning
    clean_softlabels : softversion of y_tilde
    '''
    ntrain = pred_softlabels_bar.shape[0]
    num_class = pred_softlabels_bar.shape[1]
    
    for i in range(ntrain):
        # 获取最大概率的预测类别
        m_x = pred_softlabels_bar[i].argmax()
        # 计算似然比 LR = f_y(x)/f_m_x(x)
        lr = pred_softlabels_bar[i][y_tilde[i]] / pred_softlabels_bar[i][m_x]
        # 如果 LR < δ,则翻转标签
        if lr < delta:
            y_tilde[i] = m_x

    eps = 1e-2
    clean_softlabels = torch.ones(ntrain, num_class)*eps/(num_class - 1)
    clean_softlabels.scatter_(1, torch.tensor(np.array(y_tilde)).reshape(-1, 1), 1 - eps)
    return y_tilde, clean_softlabels


def learning_rate(init, epoch):
    optim_factor = 0
    if (epoch > 200):
        optim_factor = 4
    elif(epoch > 160):
        optim_factor = 3
    elif(epoch > 120):
        optim_factor = 2
    elif(epoch > 60):
        optim_factor = 1

    return init*math.pow(0.5, optim_factor)


class ClassSpecificLRTCorrector:
    """针对特定类别的LRT标签修复器"""
    
    def __init__(self, class_idx, args, random_seed, transform_train, device, saver):
        self.class_idx = class_idx
        self.args = args
        self.random_seed = random_seed
        self.transform_train = transform_train
        self.device = device
        self.saver = saver
        self.class_name = ClassSpecificCOVID19.classes[class_idx]
        
        # 设置随机种子
        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        
        # 创建特定类别的数据集
        self.trainset = ClassSpecificCOVID19(root='./data', class_idx=class_idx, 
                                            split='train', train_ratio=1.0, 
                                            transform=transform_train)
        
        self.batch_size = 64
        self.num_workers = 1
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, 
            shuffle=True, num_workers=self.num_workers, 
            worker_init_fn=_init_fn
        )
        
        self.num_class = 4  # 总类别数保持不变
        self.in_channel = 1
        
        # 设置数据集大小
        self.ntrain = len(self.trainset)
        
        # 准备噪声标签
        self.prepare_noisy_labels()
        
        # 创建网络
        self.create_network()
        
        # 记录统计数据
        self.train_acc_record = []
        self.clean_train_acc_record = []
        self.noise_train_acc_record = []
        self.recovery_record = []
        
        # 预测软标签
        self.pred_softlabels = np.zeros([self.ntrain, args.every_n_epoch, self.num_class], dtype=float)
        
        # 矩阵A初始化
        self.A = 1/self.num_class*torch.ones(self.ntrain, self.num_class, self.num_class, requires_grad=False).float().to(device)
        self.h = np.zeros([self.ntrain, self.num_class])
        
        # 损失函数
        self.criterion_1 = ANL_CE_Loss(alpha=5.0, beta=5.0, delta=5e-5)
        self.criterion_2 = nn.NLLLoss()
        
        # 早停变量
        self.perfect_recovery = 0.95
        self.recovery_threshold = 3
        self.perfect_num = 0
        
    def prepare_noisy_labels(self):
        """准备带噪声的标签"""
        noise_level = self.args.noise_level
        noise_type = self.args.noise_type
        
        # 获取真实标签
        self.y_train = self.trainset.get_data_labels()
        self.y_train = np.array(self.y_train)
        
        if noise_type == 'none':
            self.noise_y_train = self.y_train
            self.p = np.eye(self.num_class)
            self.keep_indices = np.arange(len(self.y_train))
            
            # 为无噪声情况创建软标签
            eps = 1e-2
            n_samples = len(self.y_train)
            noise_softlabel = torch.ones(n_samples, self.num_class)*eps/(self.num_class-1)
            noise_softlabel.scatter_(1, torch.tensor(self.noise_y_train.reshape(-1, 1)), 1-eps)
            self.trainset.update_corrupted_softlabel(noise_softlabel)
        else:
            if noise_type == "uniform":
                self.noise_y_train, self.p, self.keep_indices = noisify_with_P(
                    self.y_train, nb_classes=self.num_class, 
                    noise=noise_level, random_state=self.random_seed
                )
                print(f"Class {self.class_name}: apply uniform noise")
                
            elif noise_type == "pairflip":
                self.noise_y_train, self.p, self.keep_indices = noisify_pairflip(
                    self.y_train, nb_classes=self.num_class, 
                    noise=noise_level, random_state=self.random_seed
                )
                print(f"Class {self.class_name}: apply pairflip noise")
                
            else:  # asymmetric
                self.noise_y_train, self.p, self.keep_indices = noisify_covid_asymmetric(
                    self.y_train, noise=noise_level, random_state=self.random_seed
                )
                print(f"Class {self.class_name}: apply asymmetric noise")
            
            # 更新数据集的噪声标签
            self.trainset.update_corrupted_label(self.noise_y_train)
            
            # 创建软标签
            eps = 1e-2
            n_samples = len(self.y_train)
            noise_softlabel = torch.ones(n_samples, self.num_class)*eps/(self.num_class-1)
            noise_softlabel.scatter_(1, torch.tensor(self.noise_y_train.reshape(-1, 1)), 1-eps)
            self.trainset.update_corrupted_softlabel(noise_softlabel)
            
            # 记录统计信息
            print(f"Class {self.class_name}: clean data num: {len(self.keep_indices)}")
            print(f"Class {self.class_name}: probability transition matrix:\n{self.p}")
            
            self.saver.write(f'Class {self.class_name}: clean data num: {len(self.keep_indices)}\n')
            self.saver.write(f'Class {self.class_name}: probability transition matrix:\n{self.p}\n')
            self.saver.flush()
    
    def create_network(self):
        """创建网络模型"""
        which_net = self.args.network
        
        if which_net == "cnn":
            self.net_trust = CNN9LAYER(input_channel=self.in_channel, n_outputs=self.num_class)
            self.net = CNN9LAYER(input_channel=self.in_channel, n_outputs=self.num_class)
            self.net.apply(weight_init)
            self.feature_size = 128
        elif which_net == 'net':
            self.net_trust = Net(n_channel=self.in_channel, n_classes=self.num_class)
            self.net = Net(n_channel=self.in_channel, n_classes=self.num_class)
            self.net.apply(weight_init)
            self.feature_size = 128
        elif which_net == 'toynet':
            self.net_trust = ToyNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.net = ToyNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.feature_size = 128
        elif which_net == 'resnet18':
            self.net_trust = resnet18(in_channel=self.in_channel, num_classes=self.num_class)
            self.net = resnet18(in_channel=self.in_channel, num_classes=self.num_class)
            self.feature_size = 512
        elif which_net == 'resnet34':
            self.net_trust = resnet34(in_channel=self.in_channel, num_classes=self.num_class)
            self.net = resnet34(in_channel=self.in_channel, num_classes=self.num_class)
            self.feature_size = 512
        elif which_net == 'preact_resnet18':
            self.net_trust = preact_resnet18(num_classes=self.num_class, num_input_channels=self.in_channel)
            self.net = preact_resnet18(num_classes=self.num_class, num_input_channels=self.in_channel)
            self.feature_size = 256
        elif which_net == 'preact_resnet34':
            self.net_trust = preact_resnet34(num_classes=self.num_class, num_input_channels=self.in_channel)
            self.net = preact_resnet34(num_classes=self.num_class, num_input_channels=self.in_channel)
            self.feature_size = 256
        elif which_net == 'preact_resnet101':
            self.net_trust = preact_resnet101()
            self.net = preact_resnet101()
            self.feature_size = 256
        elif which_net == 'robust':
            self.net_trust = RobustNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.net = RobustNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.feature_size = 512*7*7
        elif which_net == 'mixup_robust':
            self.net_trust = MixupRobustNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.net = MixupRobustNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.feature_size = 512*7*7
        elif which_net == 'ensemble_robust':
            self.net_trust = EnsembleRobustNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.net = EnsembleRobustNet(in_channels=self.in_channel, num_classes=self.num_class)
            self.feature_size = 512*7*7
        else:
            raise ValueError('Invalid network!')
            
        # 多GPU支持
        if len(self.args.opt_gpus) > 1:
            self.net_trust = torch.nn.DataParallel(self.net_trust)
            self.net = torch.nn.DataParallel(self.net)
            
        self.net_trust.to(self.device)
        self.net.to(self.device)
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        train_correct = 0
        train_loss = 0
        train_total = 0
        delta = 0.8 + 0.02*max(epoch - self.args.epoch_update + 1, 0)

        clean_train_correct = 0
        noise_train_correct = 0

        optimizer_trust = optim.SGD(self.net_trust.parameters(), 
                                   lr=learning_rate(self.args.lr, epoch), 
                                   weight_decay=5e-4,
                                   nesterov=True, momentum=0.9)

        self.net_trust.train()

        # 训练使用带噪声的数据
        for i, (images, labels, softlabels, indices) in enumerate(tqdm(self.trainloader, ncols=100, ascii=True, 
                                                                     desc=f"Class {self.class_name} Epoch {epoch}")):
            if images.size(0) == 1:  # 当batch size为1时，跳过，因为batch normalization
                continue

            images, labels, softlabels = images.to(self.device), labels.to(self.device), softlabels.to(self.device)
            outputs, features = self.net_trust(images)
            log_outputs = torch.log_softmax(outputs, 1).float()

            # 更新h矩阵
            if epoch >= self.args.epoch_start - 1 and (epoch - (self.args.epoch_start - 1)) % self.args.epoch_interval == 0:
                self.h[indices] = log_outputs.detach().cpu()
                
            normal_outputs = torch.softmax(outputs, 1)

            if epoch >= self.args.epoch_start:  # 使用loss_retro + loss_ce
                A_batch = self.A[indices].to(self.device)
                loss = sum([-A_batch[i].matmul(softlabels[i].reshape(-1, 1).float()).t().matmul(log_outputs[i])
                           for i in range(len(indices))]) / len(indices) + \
                       self.criterion_2(log_outputs, labels)
            else:  # 使用loss_ce
                loss = self.criterion_1(outputs, features, labels)

            optimizer_trust.zero_grad()
            loss.backward()
            optimizer_trust.step()

            # 记录预测的软标签
            if epoch >= (self.args.epoch_update - self.args.every_n_epoch):
                self.pred_softlabels[indices, epoch % self.args.every_n_epoch, :] = normal_outputs.detach().cpu().numpy()

            train_loss += loss.item()
            train_total += images.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            # 用于监控，如果没有真实标签，可以注释掉这部分
            train_label_clean = torch.tensor(self.y_train)[indices].to(self.device)
            train_label_noise = torch.tensor(self.noise_y_train)[indices].to(self.device)
            clean_train_correct += predicted.eq(train_label_clean).sum().item()
            noise_train_correct += predicted.eq(train_label_noise).sum().item()  # 相对于原始噪声标签的准确率

        # 计算准确率
        train_acc = train_correct / train_total * 100
        clean_train_acc = clean_train_correct / train_total * 100
        noise_train_acc = noise_train_correct / train_total * 100
        
        print(f"Class {self.class_name} - Train Epoch: [{epoch}/{self.args.n_epochs}] "
              f"Training Acc wrt Corrected {train_acc:.3f} "
              f"Train Acc wrt True {clean_train_acc:.3f} "
              f"Train Acc wrt Noise {noise_train_acc:.3f}")

        return train_acc, clean_train_acc, noise_train_acc
    
    def update_A_matrix(self, epoch):
        """更新A矩阵"""
        if epoch >= self.args.epoch_start - 1 and (epoch - (self.args.epoch_start - 1)) % self.args.epoch_interval == 0:
            cprint(f"Class {self.class_name} - +++++++++++++++++ Updating A +++++++++++++++++++", "magenta")
            unsolved = 0
            infeasible = 0
            y_soft = self.trainset.get_data_softlabel()

            with torch.no_grad():
                for i in tqdm(range(self.ntrain), ncols=100, ascii=True, desc=f"Class {self.class_name} Update A"):
                    try:
                        result, A_opt = updateA(y_soft[i], self.h[i], rho=0.9)
                    except:
                        self.A[i] = self.A[i]
                        unsolved += 1
                        continue

                    if (result == np.inf):
                        self.A[i] = self.A[i]
                        infeasible += 1
                    else:
                        self.A[i] = torch.tensor(A_opt)
    
    def update_labels(self, epoch):
        """更新标签"""
        if epoch >= self.args.epoch_update:
            y_tilde = self.trainset.get_data_labels()
            pred_softlabels_bar = self.pred_softlabels.mean(1)
            delta = 0.8 + 0.02*max(epoch - self.args.epoch_update + 1, 0)
            clean_labels, clean_softlabels = lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta)
            self.trainset.update_corrupted_softlabel(clean_softlabels)
            self.trainset.update_corrupted_label(clean_labels)
    
    def validate(self, epoch):
        """验证和记录结果"""
        if not (epoch % 5):
            self.net_trust.eval()

            self.train_acc_record.append(self.curr_train_acc)
            self.clean_train_acc_record.append(self.curr_clean_train_acc)
            self.noise_train_acc_record.append(self.curr_noise_train_acc)

            recovery_acc = np.sum(self.trainset.get_data_labels() == self.y_train) / self.ntrain
            self.recovery_record.append(recovery_acc)

            # 检查是否达到完美恢复率
            if recovery_acc >= self.perfect_recovery:
                self.perfect_num += 1
            else:
                self.perfect_num = 0

            cprint(f'Class {self.class_name} >> final recovery rate: {recovery_acc}', 'green')
            self.saver.write(f'Class {self.class_name} >> final recovery rate: {recovery_acc*100:.2f}%\n')
            self.saver.flush()
            
            return recovery_acc, self.perfect_num >= self.recovery_threshold
        
        return 0, False
    
    def run(self):
        """运行标签修复流程"""
        for epoch in range(self.args.n_epochs):
            # 训练一个epoch
            self.curr_train_acc, self.curr_clean_train_acc, self.curr_noise_train_acc = self.train_epoch(epoch)
            
            # 更新A矩阵
            self.update_A_matrix(epoch)
            
            # 更新标签
            self.update_labels(epoch)
            
            # 验证
            recovery_rate, early_stop = self.validate(epoch)
            
            # 早停
            if early_stop:
                cprint(f'Class {self.class_name} >> Early stopping: Perfect recovery rate achieved for {self.recovery_threshold} consecutive validations', 'red')
                self.saver.write(f'Class {self.class_name} >> Early stopping at epoch {epoch}: Perfect recovery rate achieved for {self.recovery_threshold} consecutive validations\n')
                self.saver.flush()
                break
                
        return self.trainset, recovery_rate


class MultiClassLRTSystem:
    """多类标签修复系统，负责协调四个子类标签修复器"""
    
    def __init__(self, args):
        self.args = args
        self.random_seed = args.seed
        
        # 设置随机种子
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True
        
        # 设置设备
        self.setup_device()
        
        # 创建日志
        self.create_log_file()
        
        # 创建数据增强
        self.create_transforms()
        
    def setup_device(self):
        """设置设备"""
        if self.args.n_gpus > 1:
            print("Using ",self.args.n_gpus, " GPUs")
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
        print(self.device)
        
    def create_log_file(self):
        """创建日志文件"""
        file_name = '[' + self.args.dataset + '_' + self.args.network + ']' \
                    + 'type:' + self.args.noise_type + '_' + 'noise:' + str(self.args.noise_level) + '_' \
                    + '_' + 'start:' + str(self.args.epoch_start) + '_' \
                    + 'every:' + str(self.args.every_n_epoch) + '_'\
                    + 'time:' + str(datetime.datetime.now()) + '.txt'
        log_dir = check_folder('new_logs/logs_txt_' + str(self.random_seed))
        file_name = os.path.join(log_dir, file_name)
        self.saver = open(file_name, "w")

        self.saver.write('noise type: {}\nnoise level: {}\nwhen_to_apply_epoch: {}\n'.format(
            self.args.noise_type, self.args.noise_level, self.args.epoch_start))
        self.saver.flush()
        
    def create_transforms(self):
        """创建数据增强"""
        if self.args.dataset == 'covid':
            self.transform_train = transforms.Compose([
                    transforms.Resize((224,224)), 
                    #transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5094,),(0.2532,)),
                ])
        else:
            self.transform_train = None
            
    def run(self):
        """运行多类标签修复系统"""
        print("Starting Multi-Class LRT System...")
        
        # 为每个类别创建标签修复器
        correctors = []
        for class_idx in range(4):  # COVID-19有4个类别
            corrector = ClassSpecificLRTCorrector(
                class_idx=class_idx,
                args=self.args,
                random_seed=self.random_seed,
                transform_train=self.transform_train,
                device=self.device,
                saver=self.saver
            )
            correctors.append(corrector)
            
        # 按序运行每个类别的标签修复
        fixed_datasets = []
        class_recovery_rates = []
        
        for i, corrector in enumerate(correctors):
            class_name = ClassSpecificCOVID19.classes[i]
            cprint(f"================  Starting LRT for Class {class_name}...  ================", "yellow")
            
            # 运行标签修复
            fixed_dataset, recovery_rate = corrector.run()
            fixed_datasets.append(fixed_dataset)
            class_recovery_rates.append(recovery_rate)
            
            cprint(f"================  Completed LRT for Class {class_name} with recovery rate {recovery_rate:.4f}  ================", "yellow")
            
        # 合并所有修复后的数据集
        combined_dataset = CombinedCOVID19Dataset(fixed_datasets, transform=self.transform_train)
        
        # 获取整体标签修复率
        y_train_all = []
        for i, dataset in enumerate(fixed_datasets):
            y_train_all.extend(correctors[i].y_train)
        y_train_all = np.array(y_train_all)
        
        y_fixed_all = combined_dataset.get_data_labels()
        overall_recovery_rate = np.sum(y_fixed_all == y_train_all) / len(y_train_all)
        
        # 记录每个类别和整体的标签修复率
        cprint(f"================  Final Results  ================", "red")
        for i, class_name in enumerate(ClassSpecificCOVID19.classes):
            cprint(f"Class {class_name}: Recovery Rate = {class_recovery_rates[i]:.4f}", "green")
            self.saver.write(f"Class {class_name}: Recovery Rate = {class_recovery_rates[i]:.4f}\n")
            
        cprint(f"Overall Recovery Rate = {overall_recovery_rate:.4f}", "red")
        self.saver.write(f"Overall Recovery Rate = {overall_recovery_rate:.4f}\n")
        self.saver.flush()
        
        return overall_recovery_rate, class_recovery_rates, combined_dataset
        

def main(args):
    # 创建并运行多类标签修复系统
    lrt_system = MultiClassLRTSystem(args)
    overall_rate, class_rates, combined_dataset = lrt_system.run()
    
    print(f"标签修复完成。整体修复率: {overall_rate:.4f}")
    
    return overall_rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, help='delimited list input of GPUs', type=int)
    parser.add_argument('--n_gpus', default=0, help="num of GPUS to use", type=int)
    parser.add_argument("--dataset", default='covid', help='choose dataset', type=str)
    parser.add_argument('--network', default='robust', help="network architecture", type=str)
    parser.add_argument('--noise_type', default='pairflip', help='noisy type', type=str)
    parser.add_argument('--noise_level', default=0.8, help='noisy level', type=float)
    parser.add_argument('--lr', default=1e-3, help="learning rate", type=float)
    parser.add_argument('--n_epochs', default=180, help="training epoch", type=int)
    parser.add_argument('--epoch_start', default=25, help='epoch start to introduce l_r', type=int)
    parser.add_argument('--epoch_update', default=15, help='epoch start to update labels', type=int)
    parser.add_argument('--epoch_interval', default=10, help="interval for updating A", type=int)
    parser.add_argument('--every_n_epoch', default=10, help='rolling window for estimating f(x)_t', type=int)
    parser.add_argument('--seed', default=123, help='set random seed', type=int)
    args = parser.parse_args()

    opt_gpus = [i for i in range(args.gpu, (args.gpu + args.n_gpus))]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)

    main(args)
