import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint


class Net(nn.Module):
    def __init__(self, n_channel=1, n_classes=10):
        super(Net, self).__init__()
        # 适配224x224灰度图输入
        self.conv1_1 = nn.Conv2d(n_channel, 32, 3, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1_1 = nn.BatchNorm2d(32)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2_1 = nn.MaxPool2d(2, 2)
        self.bn2_1 = nn.BatchNorm2d(128)

        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2_2 = nn.MaxPool2d(2, 2)
        self.bn2_2 = nn.BatchNorm2d(128)
        # 224->112->56->28 (3次2x2池化)
        self.fc1 = nn.Linear(28 * 28 * 128, 64)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        x1_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x1_2 = self.pool1(F.relu(self.bn1_2(self.conv1_2(x1_1))))

        x2_1 = self.pool2_1(F.relu(self.bn2_1(self.conv2_1(x1_2))))
        x2_2 = self.pool2_2(F.relu(self.bn2_2(self.conv2_2(x2_1))))
        x = x2_2.view(-1, 28 * 28 * 128)

        x2 = F.relu(self.fc1(x))
        x = self.fc2(x2)
        return x, x2


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.constant_(m.bias, 0)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True)
        )
        # 224->112->56->28->14 (4次stride=2)
        self.fc1 = nn.Linear(8*14*14, 64)
        self.fc2 = nn.Linear(64, 8*14*14)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        latent = self.fc1(x)
        x = self.fc2(latent)
        x = x.view(x.size(0), 8, 14, 14)
        x = self.decoder(x)
        return x, latent


def call_bn(bn, x):
    return bn(x)


class CNN9LAYER(nn.Module):
    def __init__(self, input_channel=1, n_outputs=10, dropout_rate=0.25):
        self.dropout_rate = dropout_rate
        super(CNN9LAYER, self).__init__()
        # 输入通道改为1，适配224x224输入
        self.c1 = nn.Conv2d(input_channel, 64, kernel_size=3, stride=1, padding=1)
        self.c2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.c4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        # 224->112->56->28->14->7 (5次2x2池化)
        self.l_c1 = nn.Linear(128 * 7 * 7, n_outputs)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x):

        inter_out = {}

        h = x
        h = self.c1(h)
        inter_out['act_fc1'] = F.relu(call_bn(self.bn1, h))
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        inter_out['act_fc2'] = F.relu(call_bn(self.bn2, h))
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 299->149

        h = self.c3(h)
        inter_out['act_fc3'] = F.relu(call_bn(self.bn3, h))
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = self.c4(h)
        inter_out['act_fc4'] = F.relu(call_bn(self.bn4, h))
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 149->74

        h = self.c5(h)
        inter_out['act_fc5'] = F.relu(call_bn(self.bn5, h))
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        inter_out['act_fc6'] = F.relu(call_bn(self.bn6, h))
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 74->37

        h = self.c7(h)
        inter_out['act_fc7'] = F.relu(call_bn(self.bn7, h))
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        inter_out['act_fc8'] = F.relu(call_bn(self.bn8, h))
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 37->18

        h = self.c9(h)
        inter_out['act_fc9'] = F.relu(call_bn(self.bn9, h))
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)  # 18->9

        h = h.view(h.size(0), -1)  # 展平成(batch, 128*7*7)
        logit = self.l_c1(h)
        return logit, h


class LSTMTiny(nn.Module):
    def __init__(self, num_class):
        super(LSTMTiny, self).__init__()
        self.num_class = num_class
        self.num_words = 5000
        self.embed_dim = 128
        self.lstm_hidden_dim = 512

        self.embed = nn.Embedding(self.num_words, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(self.lstm_hidden_dim, self.lstm_hidden_dim // 2)
        self.fc2 = nn.Linear(self.lstm_hidden_dim // 2, self.num_class)

    def forward(self, x):
        embed_x = self.embed(x)
        rnn_out, (hn, cn) = self.lstm(embed_x)

        hn = torch.transpose(hn, 0, 1).contiguous()
        hn = hn.view(hn.size(0), -1)

        feat = self.fc1(hn)
        # x = F.dropout(x)

        x = self.fc2(feat)

        return x, feat


class RobustLabelCorrector(nn.Module):
    """
    针对高噪声（0.8噪声率）设计的鲁棒标签修复模块
    仿照Net类的输入输出：输入224x224图像，输出预测logits和特征
    """
    def __init__(self, n_channel=1, n_classes=10, ensemble_size=3):
        super(RobustLabelCorrector, self).__init__()
        self.n_classes = n_classes
        self.ensemble_size = ensemble_size
        
        # 多分支特征提取器 - 提高鲁棒性
        self.branch1 = self._create_branch(n_channel)
        self.branch2 = self._create_branch(n_channel)  
        self.branch3 = self._create_branch(n_channel)
        
        # 自注意力机制 - 学习特征重要性
        self.attention = nn.MultiheadAttention(embed_dim=192, num_heads=8, dropout=0.1)
        self.attention_norm = nn.LayerNorm(192)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # 多头预测器 - 集成学习
        self.predictors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(64, n_classes)
            ) for _ in range(ensemble_size)
        ])
        
        # 不确定性估计器
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 权重融合网络
        self.weight_net = nn.Sequential(
            nn.Linear(128, ensemble_size),
            nn.Softmax(dim=1)
        )
        
        # Dropout层用于Monte Carlo dropout
        self.mc_dropout = nn.Dropout(0.5)
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _create_branch(self, n_channel):
        """创建单个特征提取分支"""
        return nn.Sequential(
            # 第一层卷积组
            nn.Conv2d(n_channel, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 224->112
            nn.Dropout2d(0.1),
            
            # 第二层卷积组
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112->56
            nn.Dropout2d(0.15),
            
            # 第三层卷积组
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56->28
            nn.Dropout2d(0.2),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((2, 2)),  # 输出2x2
            nn.Flatten(),
            nn.Linear(128 * 4, 64)  # 输出64维特征
        )
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, training=True, mc_samples=10):
        batch_size = x.size(0)
        
        # 多分支特征提取
        feat1 = self.branch1(x)  # [B, 64]
        feat2 = self.branch2(x)  # [B, 64]
        feat3 = self.branch3(x)  # [B, 64]
        
        # 拼接多分支特征
        multi_feat = torch.cat([feat1, feat2, feat3], dim=1)  # [B, 192]
        
        # 自注意力机制
        # 将特征reshape为序列格式 [seq_len, batch, embed_dim]
        feat_seq = multi_feat.unsqueeze(0)  # [1, B, 192]
        attn_feat, _ = self.attention(feat_seq, feat_seq, feat_seq)
        attn_feat = self.attention_norm(attn_feat.squeeze(0) + multi_feat)
        
        # 特征融合
        fused_feat = self.feature_fusion(attn_feat)  # [B, 128]
        
        if training:
            # 训练时使用集成学习
            predictions = []
            for predictor in self.predictors:
                pred = predictor(fused_feat)
                predictions.append(pred)
            
            # 计算ensemble权重
            weights = self.weight_net(fused_feat)  # [B, ensemble_size]
            
            # 加权融合预测结果
            ensemble_pred = torch.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                ensemble_pred += weights[:, i:i+1] * pred
            
            # 不确定性估计
            uncertainty = self.uncertainty_estimator(fused_feat)
            
            return ensemble_pred, fused_feat, uncertainty, predictions
        
        else:
            # 测试时使用Monte Carlo Dropout进行不确定性估计
            predictions = []
            for _ in range(mc_samples):
                # 应用MC dropout
                mc_feat = self.mc_dropout(fused_feat)
                
                # 集成预测
                sample_preds = []
                for predictor in self.predictors:
                    pred = predictor(mc_feat)
                    sample_preds.append(pred)
                
                # 计算权重并融合
                weights = self.weight_net(mc_feat)
                ensemble_pred = torch.zeros_like(sample_preds[0])
                for i, pred in enumerate(sample_preds):
                    ensemble_pred += weights[:, i:i+1] * pred
                
                predictions.append(ensemble_pred)
            
            # 计算均值和方差
            predictions = torch.stack(predictions, dim=0)  # [mc_samples, B, n_classes]
            mean_pred = predictions.mean(dim=0)
            pred_var = predictions.var(dim=0)
            uncertainty = pred_var.mean(dim=1, keepdim=True)  # [B, 1]
            
            return mean_pred, fused_feat, uncertainty
    
    def get_confident_predictions(self, x, confidence_threshold=0.7):
        """
        获取高置信度的预测结果，用于标签修复
        
        Args:
            x: 输入图像
            confidence_threshold: 置信度阈值
            
        Returns:
            confident_preds: 高置信度预测
            confident_mask: 置信度掩码
            uncertainties: 不确定性估计
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(x, training=False)
            pred, feat, uncertainty = result[:3]  # 只取前3个返回值
            
            # 计算预测置信度
            probs = F.softmax(pred, dim=1)
            max_probs, _ = probs.max(dim=1)
            
            # 结合不确定性和最大概率来确定置信度
            confidence = max_probs * (1 - uncertainty.squeeze())
            confident_mask = confidence > confidence_threshold
            
            return pred, confident_mask, uncertainty
    
    def correct_labels(self, images, noisy_labels, confidence_threshold=0.7):
        """
        标签修复函数
        
        Args:
            images: 输入图像
            noisy_labels: 噪声标签
            confidence_threshold: 置信度阈值
            
        Returns:
            corrected_labels: 修复后的标签
            correction_mask: 修复掩码
        """
        pred, confident_mask, uncertainty = self.get_confident_predictions(
            images, confidence_threshold
        )
        
        # 获取预测标签
        pred_labels = pred.argmax(dim=1)
        
        # 只修复高置信度且与原标签不同的样本
        correction_mask = confident_mask & (pred_labels != noisy_labels)
        
        # 创建修复后的标签
        corrected_labels = noisy_labels.clone()
        corrected_labels[correction_mask] = pred_labels[correction_mask]
        
        return corrected_labels, correction_mask




