import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RobustNet(nn.Module):
    """
    针对高噪声率（0.8）设计的鲁棒网络
    主要特性：
    1. 多头输出结构 - 增加模型容量
    2. 特征正则化 - 提高特征稳定性
    3. 自适应权重机制 - 动态调整预测权重
    4. 残差连接 - 改善梯度流动
    5. 噪声感知模块 - 自适应噪声处理
    """
    
    def __init__(self, in_channels=1, num_classes=4, dropout_rate=0.3):
        super(RobustNet, self).__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 特征提取骨干网络
        self.backbone = self._build_backbone(in_channels)
        
        # 多头分类器
        self.classifier_heads = nn.ModuleList([
            self._build_classifier_head(512*7*7, num_classes) for _ in range(3)
        ])
        
        # 自适应权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(512*7*7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 3)
            # 移除Softmax，在forward中手动应用
        )
        
        # 噪声感知模块
        self.noise_aware = NoiseAwareModule(512*7*7, num_classes)
        
        # 特征正则化
        self.feature_norm = nn.LayerNorm(512*7*7)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _build_backbone(self, in_channels):
        """构建特征提取骨干网络"""
        return nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224 -> 112
            nn.Dropout2d(0.1),
            
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112 -> 56
            nn.Dropout2d(0.15),
            
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56 -> 28
            nn.Dropout2d(0.2),
            
            # Block 4 - 增强特征表示
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28 -> 14
            nn.Dropout2d(0.25),
            
            # Block 5 - 深层特征
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14 -> 7
        )
    
    def _build_classifier_head(self, in_features, num_classes):
        """构建分类头"""
        return nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
    
    def _init_weights(self, m):
        """初始化网络权重"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=True):
        # 特征提取
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten: (batch, 512*7*7)
        
        # 特征正则化
        normalized_features = self.feature_norm(features)
        
        # 多头预测
        head_outputs = []
        for head in self.classifier_heads:
            head_outputs.append(head(normalized_features))
        
        # 计算自适应权重
        weight_logits = self.weight_net(normalized_features)
        weights = F.softmax(weight_logits, dim=1)
        
        # 加权融合预测结果
        final_output = torch.zeros_like(head_outputs[0])
        for i, output in enumerate(head_outputs):
            final_output = final_output + weights[:, i:i+1] * output
        
        # 噪声感知调整
        noise_aware_output = self.noise_aware(normalized_features, final_output)
        
        if return_features:
            return noise_aware_output, normalized_features
        else:
            return noise_aware_output


class NoiseAwareModule(nn.Module):
    """噪声感知模块 - 根据特征质量调整预测置信度"""
    
    def __init__(self, feature_dim, num_classes):
        super(NoiseAwareModule, self).__init__()
        self.confidence_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.refinement_net = nn.Sequential(
            nn.Linear(feature_dim + num_classes, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, features, predictions):
        # 计算预测置信度
        confidence = self.confidence_net(features)
        
        # 特征和预测的联合表示
        combined = torch.cat([features, predictions], dim=1)
        refinement = self.refinement_net(combined)
        
        # 根据置信度调整预测
        # 高置信度时保持原预测，低置信度时进行refinement
        adjusted_output = confidence * predictions + (1 - confidence) * refinement
        
        return adjusted_output


class MixupRobustNet(RobustNet):
    """
    集成Mixup训练的鲁棒网络
    在训练时使用Mixup增强数据鲁棒性
    """
    
    def __init__(self, in_channels=1, num_classes=4, dropout_rate=0.3, mixup_alpha=0.2):
        super(MixupRobustNet, self).__init__(in_channels, num_classes, dropout_rate)
        self.mixup_alpha = mixup_alpha
        self.training_mode = True
    
    def mixup_data(self, x, y, alpha=0.2):
        """Mixup数据增强"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size()[0]
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    
    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        """Mixup损失函数"""
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
    
    def forward(self, x, y=None, return_features=True):
        if self.training and y is not None and self.mixup_alpha > 0:
            # 训练时使用Mixup
            mixed_x, y_a, y_b, lam = self.mixup_data(x, y, self.mixup_alpha)
            output, features = super().forward(mixed_x, return_features)
            return output, features, y_a, y_b, lam
        else:
            # 推理时正常前向传播
            return super().forward(x, return_features)


class EnsembleRobustNet(nn.Module):
    """
    集成多个鲁棒网络的元网络
    通过投票机制提高高噪声下的预测准确性
    """
    
    def __init__(self, in_channels=1, num_classes=4, num_models=3):
        super(EnsembleRobustNet, self).__init__()
        self.num_models = num_models
        
        # 创建多个具有不同配置的模型
        self.models = nn.ModuleList([
            RobustNet(in_channels, num_classes, dropout_rate=0.2 + i*0.1)
            for i in range(num_models)
        ])
        
        # 元学习器 - 学习如何组合不同模型的预测
        self.meta_learner = nn.Sequential(
            nn.Linear(num_classes * num_models, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, return_features=True):
        # 获取所有模型的预测
        predictions = []
        all_features = []
        
        for model in self.models:
            pred, feat = model(x, return_features=True)
            predictions.append(pred)
            all_features.append(feat)
        
        # 拼接所有预测
        combined_preds = torch.cat(predictions, dim=1)
        
        # 元学习器最终预测
        final_output = self.meta_learner(combined_preds)
        
        # 平均特征作为返回特征
        avg_features = torch.stack(all_features).mean(dim=0)
        
        if return_features:
            return final_output, avg_features
        else:
            return final_output


def create_robust_network(network_type='robust', in_channels=1, num_classes=4, **kwargs):
    """
    创建鲁棒网络的工厂函数
    
    Args:
        network_type: 网络类型 ('robust', 'mixup', 'ensemble')
        in_channels: 输入通道数
        num_classes: 分类数
        **kwargs: 其他参数
    """
    if network_type == 'robust':
        return RobustNet(in_channels, num_classes, **kwargs)
    elif network_type == 'mixup':
        return MixupRobustNet(in_channels, num_classes, **kwargs)
    elif network_type == 'ensemble':
        return EnsembleRobustNet(in_channels, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
