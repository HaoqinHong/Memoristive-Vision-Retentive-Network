import torch as th
import torch.nn as nn
import numpy as np
import random
import os
import math

def seed_set(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.benchmark = False
    th.backends.cudnn.deterministic = True


class Attention(nn.Module):
    def __init__(self, num_head, num_feature, args):
        super().__init__()
        self.num_head = num_head
        self.args = args
        self.pos = args.pos

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


class Retention(nn.Module):
    def __init__(self, num_head, num_sequence, num_feature, args, parallel=True):
        super().__init__()
        self.num_head = num_head
        self.num_sequence = num_sequence
        self.num_feature = num_feature
        self.args = args
        self.pos = args.pos
        self.parallel = parallel

        self.queries = nn.Linear(num_feature, num_feature, bias=False)
        self.keys = nn.Linear(num_feature, num_feature, bias=False)
        self.values = nn.Linear(num_feature, num_feature, bias=False)

        self.theta = 10000**(-th.arange(0, num_feature // num_head, 1).unsqueeze(0)
                             * 2 / (num_feature // num_head)).to('cuda')
        self.index = th.arange(0, num_sequence, dtype=th.float).unsqueeze(0).T.to('cuda')
        self.theta = th.matmul(self.index, self.theta)
        self.cos = th.cos(self.theta)
        self.sin = th.sin(self.theta)

        self.mask_decay = self._mask_decay_generate().to('cuda')
        self.gamma_decay = self._gamma_decay_generate().to('cuda')

        self.norm = nn.LayerNorm(num_feature // num_head)
        self.swish = nn.SiLU()
        self.out = nn.Linear(num_feature, num_feature, bias=False)

    def _mask_decay_generate(self):
        mask = th.zeros([self.num_head, self.num_sequence, self.num_sequence])
        for k in range(self.num_head):
            gamma = 1 - 2**(-5 - k)
            for i in range(self.num_sequence):
                for j in range(i + 1):
                    mask[k, i, j] = gamma**(i - j)
        return mask

    def _gamma_decay_generate(self):
        mask = th.ones([self.num_head, self.num_feature // self.num_head, self.num_feature // self.num_head])
        for k in range(self.num_head):
            gamma = 1 - 2**(-5 - k)
            mask[k] = gamma * mask[k]
        return mask

    def forward_parallel(self, x):
        b, s, d = x.shape

        q = self.queries(x).reshape(b, s, self.num_head, d // self.num_head)
        q = q.transpose(1, 2)
        k = self.keys(x).reshape(b, s, self.num_head, d // self.num_head)
        k = k.transpose(1, 2)
        v = self.values(x).reshape(b, s, self.num_head, d // self.num_head)
        v = v.transpose(1, 2)

        q_cos = q * self.cos
        q_sin = q * self.sin

        k_cos = k * self.cos
        k_sin = k * self.sin

        r = th.matmul(q_cos, k_cos.transpose(2, 3)) + th.matmul(q_sin, k_sin.transpose(2, 3))
        r = r * self.mask_decay
        r = th.matmul(r, v)

        y = self.norm(r).transpose(1, 2)
        y = y.reshape(b, s, -1)

        x = self.swish(y)
        x = self.out(x)

        return x

    def forward_recurrent(self, x):
        b, s, d = x.shape

        si_cos = th.zeros([b, self.num_head, d // self.num_head, d // self.num_head]).to('cuda')
        si_sin = th.zeros([b, self.num_head, d // self.num_head, d // self.num_head]).to('cuda')

        for i in range(s):
            qi = self.queries(x[:, i, :].unsqueeze(1)).reshape(b, 1, self.num_head, d // self.num_head)
            qi = qi.transpose(1, 2)
            ki = self.keys(x[:, i, :].unsqueeze(1)).reshape(b, 1, self.num_head, d // self.num_head)
            ki = ki.transpose(1, 2)
            vi = self.values(x[:, i, :].unsqueeze(1)).reshape(b, 1, self.num_head, d // self.num_head)
            vi = vi.transpose(1, 2)

            qi_cos = qi * self.cos[i, :].unsqueeze(0)
            qi_sin = qi * self.sin[i, :].unsqueeze(0)

            ki_cos = ki * self.cos[i, :].unsqueeze(0)
            ki_sin = ki * self.sin[i, :].unsqueeze(0)

            si_cos = self.gamma_decay * si_cos + th.matmul(ki_cos.transpose(2, 3), vi)
            si_sin = self.gamma_decay * si_sin + th.matmul(ki_sin.transpose(2, 3), vi)

            ri = th.matmul(qi_cos, si_cos) + th.matmul(qi_sin, si_sin)

            yi = self.norm(ri).transpose(1, 2)
            yi = yi.reshape(b, 1, -1)

            xi = self.swish(yi)
            xi = self.out(xi)

            if not i:
                o = xi
            else:
                o = th.concat([o, xi], 1)

        return o

    def forward(self, x):
        if self.parallel:
            return self.forward_parallel(x)
        else:
            return self.forward_recurrent(x)


class FeedForward(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.ff1 = nn.Linear(num_feature, num_feature)
        self.ff2 = nn.Linear(num_feature, num_feature)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ff1(x)
        x = self.gelu(x)
        x = self.ff2(x)

        return x


class RetBlock(nn.Module):
    def __init__(self, num_head, num_sequence, num_feature, args, parallel=True):
        super().__init__()
        self.retention = Retention(num_head, num_sequence, num_feature, args, parallel)
        self.norm1 = nn.LayerNorm(num_feature)
        self.ffn = FeedForward(num_feature)
        self.norm2 = nn.LayerNorm(num_feature)

    def forward(self, x):
        x = self.norm1(self.retention(x) + x)
        x = self.norm2(self.ffn(x) + x)

        return x


class TransBlock(nn.Module):
    def __init__(self, num_head, num_feature, args):
        super().__init__()
        self.attention = Attention(num_head, num_feature, args)
        self.norm1 = nn.LayerNorm(num_feature)
        self.ffn = FeedForward(num_feature)
        self.norm2 = nn.LayerNorm(num_feature)

    def forward(self, x):
        x = self.norm1(self.attention(x) + x)
        x = self.norm2(self.ffn(x) + x)

        return x


class RetNet(nn.Module):
    def __init__(self, num_layer, num_head, num_sequence, num_feature, args, parallel=True):
        super().__init__()
        self.args = args
        self.pos = self.args.pos
        self.num_layer = num_layer
        self.num_head = num_head
        self.pe = PositionalEncoding(num_sequence, num_feature)
        self.encoder = nn.Sequential(*[RetBlock(num_head, num_sequence, num_feature, args, parallel)for _ in range(num_layer)])
        self.fc = nn.Linear(num_feature * num_sequence, 10, bias=False)

    def forward(self, x):
        if self.pos:
            x = self.pe(x)
        x = self.encoder(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        return x

    def info(self):
        return f'rn_{self.num_layer}_{self.num_head}_{self.pos}'


class Transformer(nn.Module):
    def __init__(self, num_layer, num_head, num_sequence, num_feature, args):
        super().__init__()
        self.args = args
        self.pos = self.args.pos
        self.num_layer = num_layer
        self.num_head = num_head
        self.pe = PositionalEncoding(num_sequence, num_feature)
        self.encoder = nn.Sequential(*[TransBlock(num_head, num_feature, args)for _ in range(num_layer)])
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


class PositionalEncoding(nn.Module):
    def __init__(self, num_sequence, num_feature):
        super().__init__()
        pe = th.tensor(np.array([[pos / np.power(10000, 2 * i / num_feature) for i in range(num_feature)]
                                 if pos != 0 else np.zeros(num_feature) for pos in range(num_sequence)]),
                       dtype=th.float32)

        pe[:, 0::2] = th.sin(pe[:, 0::2])
        pe[:, 1::2] = th.cos(pe[:, 1::2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x


def patch_transform(tensor_in, size_patch):
    # tensor_in: [batch, channel, height, width]
    unfold = nn.Unfold(kernel_size=size_patch, stride=size_patch)
    tensor_out = unfold(tensor_in)
    # tensor_out: [batch, channel*size_patch*size_patch, num_patch]
    tensor_out = tensor_out.transpose(2, 1)

    return tensor_out


if __name__ == '__main__':
    seed_set(0)

    inputs = th.ones(1, 10, 6).to('cuda')

    # model = Transformer(4, 2, 10, 6, pos=True).to('cuda')
    model = RetNet(4, 2, 10, 6, pos=True).to('cuda')
    res = model(inputs)
    # print(res)
    # print(th.linspace(0, 1, 3))
    angle = 1.0 / (10000 ** th.linspace(0, 1, 16 // 2 // 2))
    angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
    index = th.arange(0, 20, dtype=th.float).unsqueeze(0).T
    print(index*angle)
