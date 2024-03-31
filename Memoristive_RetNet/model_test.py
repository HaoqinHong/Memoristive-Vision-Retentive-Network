import torch as th
import torch.nn as nn
import numpy as np


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = th.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def patch_transform(tensor_in, size_patch):
    # tensor_in: [batch, channel, height, width]
    unfold = nn.Unfold(kernel_size=size_patch, stride=size_patch)
    tensor_out = unfold(tensor_in)
    # tensor_out: [batch, channel*size_patch*size_patch, num_patch]
    tensor_out = tensor_out.transpose(2, 1)

    return tensor_out


class Retention(nn.Module):
    def __init__(self, num_head, num_sequence, num_feature):
        super().__init__()
        self.num_head = num_head
        self.num_sequence = num_sequence
        self.num_feature = num_feature

        self.decay = th.log(1 - 2 ** (-5 - th.arange(num_head, dtype=th.float))).to('cuda')
        self.index = th.arange(num_sequence).to(self.decay)

        self.mask = self.mask_generate()
        self.sin, self.cos = self.pos_generate()

        self.queries = nn.Linear(num_feature, num_feature, bias=False)
        self.keys = nn.Linear(num_feature, num_feature, bias=False)
        self.values = nn.Linear(num_feature, num_feature, bias=False)

        self.norm = nn.LayerNorm(num_feature // num_head)
        self.swish = nn.SiLU()
        self.out = nn.Linear(num_feature, num_feature, bias=False)

    def theta_shift(self, x):
        return (x * self.cos) + (rotate_every_two(x) * self.sin)

    def mask_generate(self):
        mask = th.tril(th.ones(self.num_sequence, self.num_sequence).to(self.decay))
        mask = th.masked_fill(self.index[:, None] - self.index[None, :], ~mask.bool(), float("inf"))
        mask = th.exp(mask * self.decay[:, None, None])
        mask = th.nan_to_num(mask).to('cuda')

        return mask

    def pos_generate(self):
        angle = 1.0 / (10000 ** th.linspace(0, 1, self.num_feature // self.num_head // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten().to('cuda')
        sin = th.sin(self.index[:, None] * angle[None, :])
        cos = th.cos(self.index[:, None] * angle[None, :])

        return sin, cos

    def forward_parallel(self, x):
        b, s, d = x.shape

        q = self.queries(x).reshape(b, s, self.num_head, d // self.num_head)
        q = q.transpose(1, 2)
        k = self.keys(x).reshape(b, s, self.num_head, d // self.num_head)
        k = k.transpose(1, 2)
        v = self.values(x).reshape(b, s, self.num_head, d // self.num_head)
        v = v.transpose(1, 2)

        q = self.theta_shift(q)
        k = self.theta_shift(k)

        r = th.matmul(q, k.transpose(2, 3))
        r = r * self.mask
        r = th.matmul(r, v)

        y = self.norm(r).transpose(1, 2)
        y = y.reshape(b, s, -1)

        x = self.swish(y)
        x = self.out(x)

        return x

    def forward(self, x):
        return self.forward_parallel(x)


class RetentionGated(nn.Module):
    def __init__(self, num_head, num_sequence, num_feature):
        super().__init__()
        self.num_head = num_head
        self.num_sequence = num_sequence
        self.num_feature = num_feature

        self.decay = th.log(1 - 2 ** (-5 - th.arange(num_head, dtype=th.float))).to('cuda')
        self.index = th.arange(num_sequence).to(self.decay)

        self.mask = self.mask_generate()
        self.sin, self.cos = self.pos_generate()

        self.queries = nn.Linear(num_feature, num_feature, bias=False)
        self.keys = nn.Linear(num_feature, num_feature, bias=False)
        self.values = nn.Linear(num_feature, num_feature, bias=False)

        self.norm = nn.LayerNorm(num_feature // num_head)
        self.swish = nn.SiLU()
        self.out = nn.Linear(num_feature, num_feature, bias=False)
        self.gate = nn.Linear(num_feature, num_feature, bias=False)

    def theta_shift(self, x):
        return (x * self.cos) + (rotate_every_two(x) * self.sin)

    def mask_generate(self):
        mask = th.tril(th.ones(self.num_sequence, self.num_sequence).to(self.decay))
        mask = th.masked_fill(self.index[:, None] - self.index[None, :], ~mask.bool(), float("inf"))
        mask = th.exp(mask * self.decay[:, None, None])
        mask = th.nan_to_num(mask).to('cuda')

        return mask

    def pos_generate(self):
        angle = 1.0 / (10000 ** th.linspace(0, 1, self.num_feature // self.num_head // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten().to('cuda')
        sin = th.sin(self.index[:, None] * angle[None, :])
        cos = th.cos(self.index[:, None] * angle[None, :])

        return sin, cos

    def forward_parallel(self, x):
        b, s, d = x.shape

        q = self.queries(x).reshape(b, s, self.num_head, d // self.num_head)
        q = q.transpose(1, 2)
        k = self.keys(x).reshape(b, s, self.num_head, d // self.num_head)
        k = k.transpose(1, 2)
        v = self.values(x).reshape(b, s, self.num_head, d // self.num_head)
        v = v.transpose(1, 2)

        q = self.theta_shift(q)
        k = self.theta_shift(k)

        r = th.matmul(q, k.transpose(2, 3))
        r = r * self.mask
        r = th.matmul(r, v)

        y = self.norm(r).transpose(1, 2)
        y = y.reshape(b, s, -1)

        x = self.gate(x)
        x = self.swish(x)
        x = x * y
        x = self.out(x)

        return x

    def forward(self, x):
        return self.forward_parallel(x)


class Attention(nn.Module):
    def __init__(self, num_head, num_feature):
        super().__init__()
        self.num_head = num_head

        self.queries = nn.Linear(num_feature, num_feature, bias=False)
        self.keys = nn.Linear(num_feature, num_feature, bias=False)
        self.values = nn.Linear(num_feature, num_feature, bias=False)

        self.softmax = nn.Softmax(-1)

        self.out = nn.Linear(num_feature, num_feature, bias=False)

    def forward(self, x):
        b, s, d = x.shape

        q = self.queries(x).reshape(b, s, self.num_head, d // self.num_head)
        q = q.transpose(1, 2)
        k = self.keys(x).reshape(b, s, self.num_head, d // self.num_head)
        k = k.transpose(1, 2)
        v = self.values(x).reshape(b, s, self.num_head, d // self.num_head)
        v = v.transpose(1, 2)

        x = self.softmax(th.matmul(q, k.transpose(2, 3)) / np.sqrt(d // self.num_head))
        x = th.matmul(x, v).transpose(1, 2)
        x = x.reshape(b, s, -1)
        x = self.out(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.ff1 = nn.Linear(num_feature, num_feature, bias=False)
        self.ff2 = nn.Linear(num_feature, num_feature, bias=False)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_sequence, num_feature):
        super().__init__()
        pe = th.tensor(np.array([[pos / np.power(10000, 2 * i / num_feature) for i in range(num_feature)]
                                 if pos != 0 else np.zeros(num_feature) for pos in range(num_sequence)]), dtype=th.float32)

        pe[:, 0::2] = th.sin(pe[:, 0::2])
        pe[:, 1::2] = th.cos(pe[:, 1::2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


class RetBlock(nn.Module):
    def __init__(self, num_head, num_sequence, num_feature):
        super().__init__()
        self.retention = Retention(num_head, num_sequence, num_feature)
        self.norm1 = nn.LayerNorm(num_feature)
        self.ffn = FeedForward(num_feature)
        self.norm2 = nn.LayerNorm(num_feature)

    def forward(self, x):
        x = self.norm1(self.retention(x) + x)
        x = self.norm2(self.ffn(x) + x)

        return x


class RetBlockGated(nn.Module):
    def __init__(self, num_head, num_sequence, num_feature):
        super().__init__()
        self.retention = RetentionGated(num_head, num_sequence, num_feature)
        self.norm1 = nn.LayerNorm(num_feature)
        self.ffn = FeedForward(num_feature)
        self.norm2 = nn.LayerNorm(num_feature)

    def forward(self, x):
        x = self.norm1(self.retention(x) + x)
        x = self.norm2(self.ffn(x) + x)

        return x


class TransBlock(nn.Module):
    def __init__(self, num_head, num_feature):
        super().__init__()
        self.attention = Attention(num_head, num_feature)
        self.norm1 = nn.LayerNorm(num_feature)
        self.ffn = FeedForward(num_feature)
        self.norm2 = nn.LayerNorm(num_feature)

    def forward(self, x):
        x = self.norm1(self.attention(x) + x)
        x = self.norm2(self.ffn(x) + x)

        return x


class RetNet(nn.Module):
    def __init__(self, num_layer, num_head, num_sequence, num_feature, pos=False):
        super().__init__()
        self.pos = pos
        self.num_layer = num_layer
        self.num_head = num_head
        self.pe = PositionalEncoding(num_sequence, num_feature)
        self.encoder = nn.Sequential(*[RetBlock(num_head, num_sequence, num_feature)for _ in range(num_layer)])
        self.fc = nn.Linear(num_feature * num_sequence, 4, bias=False)

    def forward(self, x):
        if self.pos:
            x = self.pe(x)
        x = self.encoder(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        return x

    def info(self):
        return f'rn_{self.num_layer}_{self.num_head}_{self.pos}'


class RetNetGated(nn.Module):
    def __init__(self, num_layer, num_head, num_sequence, num_feature, pos=False):
        super().__init__()
        self.pos = pos
        self.num_layer = num_layer
        self.num_head = num_head
        self.pe = PositionalEncoding(num_sequence, num_feature)
        self.encoder = nn.Sequential(*[RetBlockGated(num_head, num_sequence, num_feature)for _ in range(num_layer)])
        self.fc = nn.Linear(num_feature * num_sequence, 10, bias=False)

    def forward(self, x):
        if self.pos:
            x = self.pe(x)
        x = self.encoder(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        return x

    def info(self):
        return f'rng_{self.num_layer}_{self.num_head}_{self.pos}'


class Transformer(nn.Module):
    def __init__(self, num_layer, num_head, num_sequence, num_feature, pos=False):
        super().__init__()
        self.pos = pos
        self.num_layer = num_layer
        self.num_head = num_head
        self.pe = PositionalEncoding(num_sequence, num_feature)
        self.encoder = nn.Sequential(*[TransBlock(num_head, num_feature)for _ in range(num_layer)])
        self.fc = nn.Linear(num_feature * num_sequence, 10, bias=False)

    def forward(self, x):
        if self.pos:
            x = self.pe(x)
        x = self.encoder(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        return x

    def info(self):
        return f'tn_{self.num_layer}_{self.num_head}_{self.pos}'


if __name__ == '__main__':
    # model = RetNet(2, 2, 49, 16)
    model = RetNetGated(2, 2, 49, 16).to('cuda')
    input = th.rand([10, 49, 16]).to('cuda')
    output = model(input)
    print(output.shape)
