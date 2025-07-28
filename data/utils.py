import os
import os.path
import hashlib
import errno
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_bar_updater(pbar):
    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update


def check_integrity(fpath, md5=None):
    if md5 is None:
        return True
    if not os.path.isfile(fpath):
        return False
    md5o = hashlib.md5()
    with open(fpath, 'rb') as f:
        # read in 1MB chunks
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            md5o.update(chunk)
    md5c = md5o.hexdigest()
    if md5c != md5:
        return False
    return True


def makedir_exist_ok(dirpath):
    """
    Python2 support for os.makedirs(.., exist_ok=True)
    """
    try:
        os.makedirs(dirpath)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise


def download_url(url, root, filename=None, md5=None):
    """Download a file from a url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str): Name to save the file under. If None, use the basename of the URL
        md5 (str): MD5 checksum of the download. If None, do not check
    """
    from six.moves import urllib

    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    makedir_exist_ok(root)

    # downloads file
    if os.path.isfile(fpath) and check_integrity(fpath, md5):
        print('Using downloaded and verified file: ' + fpath)
    else:
        try:
            print('Downloading ' + url + ' to ' + fpath)
            urllib.request.urlretrieve(
                url, fpath,
                reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
            )
        except OSError:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath,
                    reporthook=gen_bar_updater(tqdm(unit='B', unit_scale=True))
                )


def list_dir(root, prefix=False):
    """List all directories at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the directories found
    """
    root = os.path.expanduser(root)
    directories = list(
        filter(
            lambda p: os.path.isdir(os.path.join(root, p)),
            os.listdir(root)
        )
    )

    if prefix is True:
        directories = [os.path.join(root, d) for d in directories]

    return directories


def list_files(root, suffix, prefix=False):
    """List all files ending with a suffix at a given root

    Args:
        root (str): Path to directory whose folders need to be listed
        suffix (str or tuple): Suffix of the files to match, e.g. '.png' or ('.jpg', '.png').
            It uses the Python "str.endswith" method and is passed directly
        prefix (bool, optional): If true, prepends the path to each result, otherwise
            only returns the name of the files found
    """
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )

    if prefix is True:
        files = [os.path.join(root, d) for d in files]

    return files

# 定义ANL-CE损失函数
class ANL_CE_Loss(nn.Module):
    def __init__(self, alpha=5.0, beta=5.0, delta=5e-5, p_min=1e-7):
        """
        ANL-CE损失函数实现
        参数:
            alpha: NCE主动损失的权重 (默认5.0)
            beta: NNCE被动损失的权重 (默认5.0)
            delta: L1正则化系数 (默认5e-5)
            p_min: 概率最小值，避免log(0)问题 (默认1e-7)
        """
        super(ANL_CE_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.A = -torch.log(torch.tensor(p_min))  # 垂直翻转常数

    def forward(self, outputs, features, labels):
        """
        计算ANL-CE损失
        参数:
            outputs: 模型原始输出 (未归一化logits)
            features: 模型提取的特征 (用于L1正则化)
            labels: 真实标签
        返回:
            总损失值
        """
        # 1. 计算LogSoftmax
        log_probs = F.log_softmax(outputs, dim=1)
        
        # 2. 计算NCE主动损失 (公式3)
        nce_numerator = -log_probs.gather(1, labels.view(-1, 1))
        nce_denominator = torch.sum(-log_probs, dim=1, keepdim=True)
        nce_loss = nce_numerator / nce_denominator
        
        # 3. 计算NNCE被动损失 (公式9)
        log_probs_shifted = log_probs + self.A
        nn_denominator = torch.sum(log_probs_shifted, dim=1, keepdim=True)
        nn_numerator = log_probs_shifted.gather(1, labels.view(-1, 1))
        nn_loss = 1 - nn_numerator / nn_denominator
        
        # 4. 组合ANL损失 (公式11)
        anl_loss = self.alpha * nce_loss + self.beta * nn_loss
        
        # 5. 添加L1正则化 (针对特征张量)
        l1_reg = torch.norm(features, p=1)
        
        # 6. 总损失
        total_loss = anl_loss.mean() + self.delta * l1_reg
        
        return total_loss

    
