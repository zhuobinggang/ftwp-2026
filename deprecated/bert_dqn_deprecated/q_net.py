import torch
import torch.nn as nn
from datetime import datetime

def get_time_str():
    dt = datetime.now()
    # %f 直接返回 6 位微秒数
    return dt.strftime('%Y%m%d_%H%M%S_%f')

class ChoiceTextQnet(nn.Module):
    def __init__(self, roberta_hidden_size=768, hidden_dim=256):
        """
        参数:
          - roberta_hidden_size: RoBERTa 输出的 CLS 向量维度，通常基线模型是 768
          - hidden_dim: MLP 隐藏层的维度，用于过渡特征
        """
        super().__init__()
        
        # 定义一个简单的多层感知机 (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(roberta_hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim), # 稳定微调时的梯度
            nn.ReLU(),
            nn.Dropout(0.1),          # 防止过拟合
            nn.Linear(hidden_dim, 1)  # 最终输出一个标量 Q 值
        )

    def forward(self, cls_embeddings):
        """
        输入:
          - cls_embeddings: RoBERTa 对“历史 + Obs + 某个Action”编码后提取出的 [CLS] 向量
                            形状为: [768] (无Batch) 或 [B, 768] (有Batch)
        输出:
          - q_values: 该 Action 对应的预测 Q 值 (标量或标量张量)
        """
        q_values = self.mlp(cls_embeddings)
        return q_values # batch, 1
    
    def save_checkpoint(self, base_path = 'checkpoints/q_net', name = None):
        # path = f'{base_path}/{self.prefix}_epoch_{epoch}.pth'
        if name is None:
            name = get_time_str() + '_q_net.pth'
        path = f'{base_path}/{name}'
        torch.save({
            'state': self.state_dict(),
        }, path)
    def load_checkpoint(self, path):
        # self.init_bert() # NOTE: 需要先初始化然后加载
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.load_state_dict(checkpoint['state'])
        # self.valid_score = checkpoint.get('valid_score', -1)
        # self.stop_epoch = checkpoint.get('epoch', -1)