import os
import sys
import torch.nn as nn
import torch.nn.parallel
import random
import argparse
from network.net import weight_init, CNN9LAYER ,Net
from network.resnet import resnet18, resnet34
from network.preact_resnet import preact_resnet18, preact_resnet34, preact_resnet101, initialize_weights, conv_init
from network.toynet import ToyNet
from network.robust_net import RobustNet, MixupRobustNet, EnsembleRobustNet
from data.covid_prepare import COVID19
from data.utils import ANL_CE_Loss
from data.loss import (
    MAE_Loss, MSE_Loss, CE_Loss, RCE_Loss, NCE_Loss, NNCE_Loss,
    SCE_Loss, GCE_Loss, FL_Loss, NFL_Loss, NNFL_Loss,
    AGCE_Loss, AUE_Loss, AExp_Loss,
    NCE_MAE_Loss, NCE_RCE_Loss, NFL_MAE_Loss, NFL_RCE_Loss,
    ANL_NCE_Loss, ANL_FL_Loss
)
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from Utils.noise import noisify_with_P, noisify_pairflip, noisify_covid_asymmetric
import numpy as np
import copy
from tqdm import tqdm
import pickle as pkl
from termcolor import cprint
import datetime
import math

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

def lrt_flip_scheme(pred_softlabels_bar, y_tilde, delta ):
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

def main(args):
    # 使用类特定的LRT系统替代原有的整体数据集处理方法
    from ClassLRTcorrect import MultiClassLRTSystem
    
    # 设置随机种子
    random_seed = args.seed
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    
    print('Using {}\nTest on {}\nRandom Seed {}\nevery n epoch {}\nStart at epoch {}'.
          format(args.network, args.dataset, random_seed, args.every_n_epoch, args.epoch_start))
    
    # 创建并运行多类别LRT系统，处理四个子类别数据集
    cprint("================  Starting Class-Specific LRT Correction  ================", "yellow")
    
    # 创建多类别LRT系统
    lrt_system = MultiClassLRTSystem(args)
    
    # 运行系统，对每个类别分别进行标签修复，然后合并结果
    overall_rate, class_rates, combined_dataset = lrt_system.run()
    
    cprint(f"================  Completed with Overall Recovery Rate: {overall_rate:.4f}  ================", "green")
    
    return overall_rate



if __name__ == "__main__":
    '''
    将调用分类标签修复系统执行标签修复
    ''' 

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=0, help='delimited list input of GPUs', type=int)
    parser.add_argument('--n_gpus', default=0, help="num of GPUS to use", type=int)
    parser.add_argument("--dataset", default='covid', help='choose dataset', type=str)
    parser.add_argument('--network', default='resnet18', help="network architecture", type=str)
    parser.add_argument('--noise_type', default='uniform', help='noisy type', type=str)
    parser.add_argument('--noise_level', default=0.2, help='noisy level', type=float)
    parser.add_argument('--lr', default=1e-3, help="learning rate", type=float)
    parser.add_argument('--n_epochs', default=180, help="training epoch", type=int)
    parser.add_argument('--epoch_start', default=25, help='epoch start to introduce l_r', type=int)
    parser.add_argument('--epoch_update', default=15, help='epoch start to update labels', type=int)
    parser.add_argument('--epoch_interval', default=10, help="interval for updating A", type=int)
    parser.add_argument('--every_n_epoch', default=10, help='rolling window for estimating f(x)_t', type=int)
    parser.add_argument('--seed', default=123, help='set random seed', type=int)
    args = parser.parse_args()

    # 设置GPU
    opt_gpus = [i for i in range(args.gpu, (args.gpu + args.n_gpus))]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)

    # 调用修改后的多类标签修复系统
    main(args)

