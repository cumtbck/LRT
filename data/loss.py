import math
import torch
import torch.nn.functional as F

class LabelCorrectionLossAdapter(torch.nn.Module):
    """
    标签修复训练损失函数适配器
    用于将标准损失函数适配到标签修复训练的接口
    """
    def __init__(self, base_loss, support_features=True, support_model=False):
        super(LabelCorrectionLossAdapter, self).__init__()
        self.base_loss = base_loss
        self.support_features = support_features
        self.support_model = support_model
    
    def forward(self, pred, labels, features=None, model=None):
        """
        通用接口，适配不同的损失函数调用方式
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选)
            model: 模型实例 (可选)
        """
        # 检查base_loss的forward方法支持的参数
        import inspect
        sig = inspect.signature(self.base_loss.forward)
        params = list(sig.parameters.keys())[1:]  # 排除self参数
        
        # 构建调用参数
        call_args = [pred, labels]
        call_kwargs = {}
        
        if 'features' in params and features is not None:
            call_kwargs['features'] = features
        if 'model' in params and model is not None:
            call_kwargs['model'] = model
        
        return self.base_loss(*call_args, **call_kwargs)

class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred =  F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return mae.mean()

class MeanSquareError(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mse = torch.nn.MSELoss()
    
    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        mse = self.mse(pred, label_one_hot)
        return mse

class CrossEntropy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
    
    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        ce = self.ce(pred, labels)
        return ce

class RevserseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return rce.mean()

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7):
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q

    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return gce.mean()

class SymmetricCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, alpha, beta):
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
    
class NormalizedNegativeCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, min_prob) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.A = - torch.tensor(min_prob).log()
    
    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log() # - log(1e-7) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()

class FocalLoss(torch.nn.Module):
    '''
        https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    '''

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, features=None):
        """
        标签修复训练兼容版本
        Args:
            input: 模型预测输出 (logits)
            target: 标签
            features: 特征 (可选，用于兼容)
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, num_classes=10):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, input, target, features=None):
        """
        标签修复训练兼容版本
        Args:
            input: 模型预测输出 (logits)
            target: 标签
            features: 特征 (可选，用于兼容)
        """
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss / normalizor

        return loss.mean()

class NormalizedNegativeFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma, min_prob=1e-7) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.logmp = torch.tensor(self.min_prob).log()
        self.A = - (1 - min_prob)**gamma * self.logmp
    
    def forward(self, input, target, features=None):
        """
        标签修复训练兼容版本
        Args:
            input: 模型预测输出 (logits)
            target: 标签
            features: 特征 (可选，用于兼容)
        """
        logmp = self.logmp.to(input.device)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1).clamp(min=logmp)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = 1 - (self.A - loss) / (self.num_classes * self.A - normalizor)
        return loss.mean()

class AGCELoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=1, q=2):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class AUELoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q

    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean()

class ANormLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=1.5, p=0.9):
        super(ANormLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.p = p

    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-5, max=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.sum(torch.pow(torch.abs(self.a * label_one_hot-pred), self.p), dim=1) - (self.a-1)**self.p
        return loss.mean() / self.p

class AExpLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=3):
        super(AExpLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a

    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a)
        return loss.mean()

class ActivePassiveLoss(torch.nn.Module):
    def __init__(self, active_loss, passive_loss,
                 alpha=1., beta=1.) -> None:
        super(ActivePassiveLoss, self).__init__()
        self.active_loss = active_loss
        self.passive_loss = passive_loss
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, pred, labels, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            features: 特征 (可选，用于兼容)
        """
        return self.alpha * self.active_loss(pred, labels, features) \
            + self.beta * self.passive_loss(pred, labels, features)

class ActiveNegativeLoss(torch.nn.Module):
    def __init__(self, active_loss, negative_loss,
                 alpha=1., beta=1., delta=0.) -> None:
        super().__init__()
        self.active_loss = active_loss
        self.negative_loss = negative_loss
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
    
    def forward(self, pred, labels, model=None, features=None):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            model: 模型实例 (用于L1正则化)
            features: 特征 (可选，用于兼容)
        """
        al = self.active_loss(pred, labels, features)
        nl = self.negative_loss(pred, labels, features)
        
        # 如果提供了model参数，计算L1正则化
        if model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = self.alpha * al + self.beta * nl + self.delta * l1_norm
        else:
            loss = self.alpha * al + self.beta * nl
        
        return loss

class ANL_CE_ER(torch.nn.Module):
    def __init__(self, num_classes, alpha, beta, delta, lamb, min_prob=1e-7, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.min_prob = min_prob
        self.lamb = lamb
        self.A = - torch.tensor(min_prob).log()

    def forward(self, pred, labels, model=None, features=None, **kwargs):
        """
        标签修复训练兼容版本
        Args:
            pred: 模型预测输出 (logits)
            labels: 标签
            model: 模型实例 (用于L1正则化)
            features: 特征 (可选，用于兼容)
        """
        loss_nce = self.nce(pred, labels)
        loss_nnce = self.nnce(pred, labels)
        entropy = self.entropy_reg(pred)
        
        total_loss = self.alpha * loss_nce + self.beta * loss_nnce + self.lamb * entropy
        
        # 如果提供了model参数，添加L1正则化
        if model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            total_loss += self.delta * l1_norm
            
        return total_loss
    
    def nce(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
    
    def nnce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1-self.min_prob)
        pred = self.A + pred.log() # - log(1e-7) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()
    
    def entropy_reg(self, pred):
        prob = F.softmax(pred, dim=1).clamp(min=self.min_prob, max=1-self.min_prob)
        prob_class = prob.sum(dim=0).view(-1) / prob.sum()
        prob_class = prob_class.clamp(min=self.min_prob, max=1-self.min_prob)
        entropy = math.log(self.num_classes) + (prob_class * prob_class.log()).sum()
        return entropy

# Help Function

def _apl(active_loss, passive_loss, config):
    return ActivePassiveLoss(active_loss,
                             passive_loss,
                             config['alpha'],
                             config['beta'])

def _anl(active_loss, negative_loss, config):
    return ActiveNegativeLoss(active_loss,
                              negative_loss,
                              config['alpha'],
                              config['beta'],
                              config['delta'])

# Loss

def mae(num_classes):
    return MeanAbsoluteError(num_classes)

def ce():
    return CrossEntropy()

def rce(num_classes):
    return RevserseCrossEntropy(num_classes)

def nce(num_classes):
    return NormalizedCrossEntropy(num_classes)

def sce(num_classes, config):
    return SymmetricCrossEntropy(num_classes, config['alpha'], config['beta'])

def gce(num_classes, config):
    return GeneralizedCrossEntropy(num_classes, config['q'])

def fl(config):
    return FocalLoss(gamma=config['gamma'])

def nfl(num_classes, config):
    return NormalizedFocalLoss(config['gamma'], num_classes)

def nnfl(num_classes, config):
    return NormalizedNegativeFocalLoss(num_classes, config['gamma'], config['min_prob'])

def nnce(num_classes, config):
    return NormalizedNegativeCrossEntropy(num_classes, config['min_prob'])

def agce(num_classes, config):
    return AGCELoss(num_classes, config['a'], config['q'])

def aul(num_classes, config):
    return AUELoss(num_classes, config['a'], config['q'])

def ael(num_classes, config):
    return AExpLoss(num_classes, config['a'])

# Active Passive Loss

def nce_mae(num_classes, config):
    return _apl(nce(num_classes), mae(num_classes), config)

def nce_rce(num_classes, config):
    return _apl(nce(num_classes), rce(num_classes), config)

def nfl_mae(num_classes, config):
    return _apl(nfl(num_classes, config), mae(num_classes), config)

def nfl_rce(num_classes, config):
    return _apl(nfl(num_classes, config), rce(num_classes), config)

# Asymmetric Loss

def nce_agce(num_classes, config):
    return _apl(nce(num_classes), agce(num_classes, config), config)

def nce_aul(num_classes, config):
    return _apl(nce(num_classes), aul(num_classes, config), config)

def nce_ael(num_classes, config):
    return _apl(nce(num_classes), ael(num_classes, config), config)

# Active Negative Loss

def anl_ce(num_classes, config):
    return _anl(nce(num_classes), nnce(num_classes, config), config)

def anl_fl(num_classes, config):
    return _anl(nfl(num_classes, config), nnfl(num_classes, config), config)

# Active Negative Loss with Entropy Regularization
def anl_ce_er(num_classes, config):
    return ANL_CE_ER(num_classes, config['alpha'], config['beta'],
                     config['delta'], config['lamb'], config['min_prob'])

# 简化的损失函数类，仿照ANL_CE_Loss的使用方式
# Simplified Loss Classes for One-line Definition

class MAE_Loss(torch.nn.Module):
    """平均绝对误差损失 - 单行定义版本"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.loss_fn = MeanAbsoluteError(num_classes)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class MSE_Loss(torch.nn.Module):
    """均方误差损失 - 单行定义版本"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.loss_fn = MeanSquareError(num_classes)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class CE_Loss(torch.nn.Module):
    """交叉熵损失 - 单行定义版本"""
    def __init__(self):
        super().__init__()
        self.loss_fn = CrossEntropy()
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class RCE_Loss(torch.nn.Module):
    """反向交叉熵损失 - 单行定义版本"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.loss_fn = RevserseCrossEntropy(num_classes)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NCE_Loss(torch.nn.Module):
    """归一化交叉熵损失 - 单行定义版本"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.loss_fn = NormalizedCrossEntropy(num_classes)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NNCE_Loss(torch.nn.Module):
    """归一化负交叉熵损失 - 单行定义版本"""
    def __init__(self, num_classes=10, min_prob=1e-7):
        super().__init__()
        self.loss_fn = NormalizedNegativeCrossEntropy(num_classes, min_prob)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class SCE_Loss(torch.nn.Module):
    """对称交叉熵损失 - 单行定义版本"""
    def __init__(self, num_classes=10, alpha=0.1, beta=1.0):
        super().__init__()
        self.loss_fn = SymmetricCrossEntropy(num_classes, alpha, beta)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class GCE_Loss(torch.nn.Module):
    """广义交叉熵损失 - 单行定义版本"""
    def __init__(self, num_classes=10, q=0.7):
        super().__init__()
        self.loss_fn = GeneralizedCrossEntropy(num_classes, q)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class FL_Loss(torch.nn.Module):
    """Focal损失 - 单行定义版本"""
    def __init__(self, gamma=2, alpha=None):
        super().__init__()
        self.loss_fn = FocalLoss(gamma, alpha)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NFL_Loss(torch.nn.Module):
    """归一化Focal损失 - 单行定义版本"""
    def __init__(self, num_classes=10, gamma=2):
        super().__init__()
        self.loss_fn = NormalizedFocalLoss(gamma, num_classes)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NNFL_Loss(torch.nn.Module):
    """归一化负Focal损失 - 单行定义版本"""
    def __init__(self, num_classes=10, gamma=2, min_prob=1e-7):
        super().__init__()
        self.loss_fn = NormalizedNegativeFocalLoss(num_classes, gamma, min_prob)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class AGCE_Loss(torch.nn.Module):
    """AGCE损失 - 单行定义版本"""
    def __init__(self, num_classes=10, a=1, q=2):
        super().__init__()
        self.loss_fn = AGCELoss(num_classes, a, q)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class AUE_Loss(torch.nn.Module):
    """AUE损失 - 单行定义版本"""
    def __init__(self, num_classes=10, a=1.5, q=0.9):
        super().__init__()
        self.loss_fn = AUELoss(num_classes, a, q)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class AExp_Loss(torch.nn.Module):
    """AExp损失 - 单行定义版本"""
    def __init__(self, num_classes=10, a=3):
        super().__init__()
        self.loss_fn = AExpLoss(num_classes, a)
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NCE_MAE_Loss(torch.nn.Module):
    """NCE+MAE组合损失 - 单行定义版本"""
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0):
        super().__init__()
        self.loss_fn = ActivePassiveLoss(
            NormalizedCrossEntropy(num_classes),
            MeanAbsoluteError(num_classes),
            alpha, beta
        )
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NCE_RCE_Loss(torch.nn.Module):
    """NCE+RCE组合损失 - 单行定义版本"""
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0):
        super().__init__()
        self.loss_fn = ActivePassiveLoss(
            NormalizedCrossEntropy(num_classes),
            RevserseCrossEntropy(num_classes),
            alpha, beta
        )
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NFL_MAE_Loss(torch.nn.Module):
    """NFL+MAE组合损失 - 单行定义版本"""
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0, gamma=2):
        super().__init__()
        self.loss_fn = ActivePassiveLoss(
            NormalizedFocalLoss(gamma, num_classes),
            MeanAbsoluteError(num_classes),
            alpha, beta
        )
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class NFL_RCE_Loss(torch.nn.Module):
    """NFL+RCE组合损失 - 单行定义版本"""
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0, gamma=2):
        super().__init__()
        self.loss_fn = ActivePassiveLoss(
            NormalizedFocalLoss(gamma, num_classes),
            RevserseCrossEntropy(num_classes),
            alpha, beta
        )
    
    def forward(self, outputs, features, labels):
        return self.loss_fn(outputs, labels, features)

class ANL_NCE_Loss(torch.nn.Module):
    """主动负学习NCE损失 - 单行定义版本"""
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0, delta=0.0, min_prob=1e-7):
        super().__init__()
        self.loss_fn = ActiveNegativeLoss(
            NormalizedCrossEntropy(num_classes),
            NormalizedNegativeCrossEntropy(num_classes, min_prob),
            alpha, beta, delta
        )
    
    def forward(self, outputs, features, labels, model=None):
        return self.loss_fn(outputs, labels, model, features)

class ANL_FL_Loss(torch.nn.Module):
    """主动负学习Focal损失 - 单行定义版本"""
    def __init__(self, num_classes=10, alpha=1.0, beta=1.0, delta=0.0, gamma=2, min_prob=1e-7):
        super().__init__()
        self.loss_fn = ActiveNegativeLoss(
            NormalizedFocalLoss(gamma, num_classes),
            NormalizedNegativeFocalLoss(num_classes, gamma, min_prob),
            alpha, beta, delta
        )
    
    def forward(self, outputs, features, labels, model=None):
        return self.loss_fn(outputs, labels, model, features)

# 重新定义ANL_CE_Loss以保持一致性
class ANL_CE_Loss(torch.nn.Module):
    """主动负学习交叉熵损失 - 单行定义版本（重新定义以保持一致性）"""
    def __init__(self, alpha=5.0, beta=5.0, delta=5e-5, min_prob=1e-7):
        super().__init__()
        self.loss_fn = ANL_CE_ER(10, alpha, beta, delta, 0.0, min_prob)  # lamb=0.0表示不使用熵正则化
    
    def forward(self, outputs, features, labels, model=None):
        return self.loss_fn(outputs, labels, model, features)