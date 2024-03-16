import torch
import torch.nn as nn
import torch.nn.functional as F

# B -> Batch Size
# C -> Number of Input Channels
# IH -> Image Height
# IW -> Image Width
# P -> Patch Size
# E -> Embedding Dimension
# S -> Sequence Length = IH/P * IW/P
# Q -> Query Sequence length
# K -> Key Sequence length
# V -> Value Sequence length (same as Key length)
# H -> Number of heads
# HE -> Head Embedding Dimension = E/H

# 位置编码
class EmbedLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.pos = args.pos
        self.conv1 = nn.Conv2d(args.n_channels, args.embed_dim, kernel_size=args.patch_size, stride=args.patch_size)  # Pixel Encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.embed_dim), requires_grad=True)  # Cls Token
        self.pos_embedding = nn.Parameter(torch.zeros(1, (args.img_size // args.patch_size) ** 2 + 1, args.embed_dim), requires_grad=True)  # Positional Embedding

    def forward(self, x):
        # print("x shape:", x.shape) # x shape: torch.Size([128, 1, 28, 28])
        x = self.conv1(x)  # B C IH IW -> B E IH/P IW/P (Embedding the patches)
        # print("x shape:", x.shape) # x shape: torch.Size([128, 96, 7, 7])
        x = x.reshape([x.shape[0], self.args.embed_dim, -1])  # B E IH/P IW/P -> B E S (Flattening the patches)
        x = x.transpose(1, 2)  # B E S -> B S E 
        x = torch.cat((torch.repeat_interleave(self.cls_token, x.shape[0], 0), x), dim=1)  # Adding classification token at the start of every sequence

        if self.args.pos:
            x = x + self.pos_embedding  # Adding positional encoding
        return x


# Rentention 的并行表示
class SelfParallelRetention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_attention_heads = args.n_attention_heads
        self.embed_dim = args.embed_dim
        self.head_embed_dim = self.embed_dim // self.n_attention_heads

        self.queries = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.keys = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        self.values = nn.Linear(self.embed_dim, self.head_embed_dim * self.n_attention_heads, bias=True)
        # Assume decay_mask is a provided tensor with shape [num_head, len, len]
        self.decay_mask = self._create_decay_mask()

    def _create_decay_mask(self):
        # 计算序列长度（patch的数量）
        # seq_len = (self.args.img_size // self.args.patch_size) ** 2
        seq_len = 50

        # 创建基于位置差异的衰减掩码
        decay_mask = torch.ones((self.n_attention_heads, seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                decay_mask[:, i, j] = self._decay_function(i, j)
        return decay_mask

    def _decay_function(self, i, j):
        # 一个简单的衰减函数，可以根据需要进行修改
        return 1.0 / (1.0 + abs(i - j))
    
    # def _create_decay_mask(self):
    #         # 使用50作为序列长度示例，实际中可以替换为 self.num_sequence
    #         seq_len = 50  # 或者 self.num_sequence
    #         gamma = 0.9   # 或者 self.gamma

    #         # 创建基于位置差异的衰减掩码，初始化为全1
    #         decay_mask = torch.ones((self.n_attention_heads, seq_len, seq_len))
            
    #         # 应用新的衰减函数逻辑
    #         for i in range(seq_len):
    #             for j in range(i + 1):  # 仅考虑下三角矩阵
    #                 decay_value = gamma ** (i - j)
    #                 decay_mask[:, i, j] = decay_value
    #                 # 对称地填充上三角矩阵
    #                 decay_mask[:, j, i] = decay_value

    #         return decay_mask  


    def forward(self, x):
        m, s, e = x.shape

        xq = self.queries(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)
        xk = self.keys(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)
        xv = self.values(x).reshape(m, s, self.n_attention_heads, self.head_embed_dim).transpose(1, 2)

        x = ParallelRetention(xq, xk, xv, self.decay_mask)
        x = x.reshape(m, s, e)
        return x
    
def ParallelRetention(q, k, v, decay_mask):
    retention = q @ k.transpose(-1, -2) 
    # print("retention shape:", retention.shape)
    # print("decay_mask shape:", decay_mask.shape)

    # decay_mask_resized = F.interpolate(decay_mask.unsqueeze(0), size=(50, 50), mode='bilinear', align_corners=False)
    # decay_mask_resized = decay_mask_resized.squeeze(0)

    # 假设 retention 在 GPU 上
    if retention.is_cuda:
        decay_mask = decay_mask.to(retention.device)

    retention = retention * decay_mask
    output = retention @ v + k
    output = F.group_norm(output, num_groups=output.shape[1])  # Assuming num_groups = num_heads
    return output


# 多头注意力机制
class Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention = SelfParallelRetention(args)
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim * args.forward_mul)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(args.embed_dim * args.forward_mul, args.embed_dim)
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)

    def forward(self, x):
        x = x + self.attention(self.norm1(x)) # Skip connections
        x = x + self.fc2(self.activation(self.fc1(self.norm2(x))))  # Skip connections
        return x


# 全连接分类器
class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(args.embed_dim, args.n_classes)

    def forward(self, x):
        x = x[:, 0, :]  # Get CLS token
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
# 池化后的再进行全连接的分类器
class PoolingClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fc1 = nn.Linear(args.embed_dim, args.embed_dim)
        self.activation = nn.Tanh()
        self.fc2 = nn.Linear(args.embed_dim, args.n_classes)

        # 添加一个自适应 1D 平均池化层
        self.pool = nn.AdaptiveAvgPool1d(1)  # 池化到长度为 1

    def forward(self, x):
        # 假设 x 的形状是 [batch_size, seq_length, embed_dim]
        batch_size, seq_length, embed_dim = x.shape

        # 应用 1D 池化，池化后的形状是 [batch_size, embed_dim, 1]
        # print("x shape:", x.shape)
        x = x.transpose(1, 2)  # 交换 seq_length 和 embed_dim 维度
        x = self.pool(x)  # 池化
        x = x.squeeze(2)  # 去掉长度为 1 的维度，形状变为 [batch_size, embed_dim]

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        # print("x shape:", x.shape)
        return x


# # 单层网络
# class VisionRetentive(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.embedding = EmbedLayer(args)
#         self.encoder = nn.Sequential(*[Encoder(args) for _ in range(args.n_layers)], nn.LayerNorm(args.embed_dim))
#         self.norm = nn.LayerNorm(args.embed_dim) # Final normalization layer after the last block
#         self.pooling = args.pooling

#         if self.pooling:
#             # 池化全连接分类器
#             self.classifier = PoolingClassifier(args)
#         else:
#             # 全连接分类器
#             self.classifier = Classifier(args)

#     def forward(self, x):
#         x = self.embedding(x)
#         x = self.encoder(x)
#         x = self.norm(x)
#         x = self.classifier(x)
#         return x


# 多层网络
class VisionRetentive(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embedding = EmbedLayer(args)
        # 增加编码器层数，例如，从原先的6层增加到120层
        self.encoder = nn.Sequential(*[Encoder(args) for _ in range(120)], nn.LayerNorm(args.embed_dim))
        self.norm = nn.LayerNorm(args.embed_dim) # 在最后一个模块后的最终规范化层
        self.pooling = args.pooling

        if self.pooling:
            # 池化全连接分类器
            self.classifier = PoolingClassifier(args)
        else:
            # 全连接分类器
            self.classifier = Classifier(args)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.norm(x)
        x = self.classifier(x)
        return x