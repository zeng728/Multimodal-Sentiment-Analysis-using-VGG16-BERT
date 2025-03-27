import torch
import torch.nn as nn
from transformers import BertModel
from torchvision import models

class CoAttention(nn.Module):
    """双向协同注意力模块"""
    def __init__(self, text_dim=768, image_dim=256, hidden_dim=512):
        super(CoAttention, self).__init__()
        # 文本到图像的注意力投影层
        self.text_to_image_proj = nn.Linear(text_dim, hidden_dim)
        self.image_to_text_proj = nn.Linear(image_dim, hidden_dim)
        
        # 图像到文本的注意力投影层
        self.image_to_text_proj_query = nn.Linear(image_dim, hidden_dim)
        self.text_to_image_proj_key = nn.Linear(text_dim, hidden_dim)
        
        # 注意力缩放因子
        self.scale = hidden_dim ** -0.5

    def forward(self, text_features, image_features):
        """
        Args:
            text_features: [batch_size, seq_len, text_dim]
            image_features: [batch_size, image_dim]
        Returns:
            attended_text: [batch_size, text_dim]
            attended_image: [batch_size, image_dim]
        """
        # 维度对齐（将图像特征扩展到与文本相同维度）
        image_features = image_features.unsqueeze(1)  # [batch, 1, image_dim]
        
        # 文本到图像注意力
        text_query = self.text_to_image_proj(text_features)  # [batch, seq_len, hidden]
        image_key = self.image_to_text_proj(image_features)   # [batch, 1, hidden]
        image_attn = torch.bmm(text_query, image_key.transpose(1,2)) * self.scale
        image_attn_weights = torch.softmax(image_attn, dim=1)
        attended_image = torch.bmm(image_attn_weights.transpose(1,2), text_features).squeeze(1)

        # 图像到文本注意力
        image_query = self.image_to_text_proj_query(image_features)  # [batch, 1, hidden]
        text_key = self.text_to_image_proj_key(text_features)        # [batch, seq_len, hidden]
        text_attn = torch.bmm(image_query, text_key.transpose(1,2)) * self.scale
        text_attn_weights = torch.softmax(text_attn, dim=2)
        attended_text = torch.bmm(text_attn_weights, text_features).squeeze(1)

        return attended_text, attended_image

class FusionModel(nn.Module):
    def __init__(self, model_name, num_classes, dropout_prob=0.5):
        super(FusionModel, self).__init__()
        self.TextModel = TextModel(model_name=model_name)
        self.ImageModel = ImageModel()
        
        # 协同注意力模块
        self.co_attention = CoAttention()
        
        # 修改后的融合层
        self.fusion_fc = nn.Sequential(
            nn.Linear(768 + 256, 512),  # 原始特征拼接
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(512, num_classes)
        )
        self.to('cuda')

    def forward(self, x):
        # 提取原始特征
        image_features = self.ImageModel(x['image'])  # [batch, 256]
        text_features = self.TextModel(x['input_ids'])  # [batch, 768]
        
        # 扩展文本特征维度（假设取[CLS]标记）
        text_features = text_features.unsqueeze(1)  # [batch, 1, 768]
        
        # 协同注意力计算
        attended_text, attended_image = self.co_attention(text_features, image_features)
        
        # 特征拼接（注意力后的特征 + 原始特征）
        fused_features = torch.cat([
            attended_text, 
            attended_image,
            text_features.squeeze(1),
            image_features
        ], dim=1)  # [batch, 768+256+768+256]
        
        # 分类预测
        y_pred = self.fusion_fc(fused_features)
        return y_pred

# 原ImageModel和TextModel保持不变
