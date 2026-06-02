import torch
import torch.nn as nn

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