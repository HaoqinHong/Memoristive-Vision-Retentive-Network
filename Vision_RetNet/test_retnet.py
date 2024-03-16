from numpy import dtype
import torch as th
import torch.nn as nn


class ParallelRetention(nn.Module):
    def __init__(self, num_sequence, num_feature, gamma):
        super().__init__()
        self.queries = nn.Linear(num_feature, num_feature)
        self.keys = nn.Linear(num_feature, num_feature)
        self.values = nn.Linear(num_feature, num_feature)

        self.theta = nn.Parameter(th.rand([1, num_feature]))
        self.index = th.arange(1, num_sequence + 1, dtype=th.float).unsqueeze(0).T
        self.mask_theta = th.matmul(self.index, self.theta)
        self.mask_cos = th.cos(self.mask_theta)
        self.mask_sin = th.sin(self.mask_theta)

        self.mask_decay = self._mask_decay_generate(gamma, num_sequence)

    def _mask_decay_generate(self, gamma, num_sequence):
        mask = th.zeros([num_sequence, num_sequence])
        for i in range(num_sequence):
            for j in range(i + 1):
                mask[i, j] = gamma**(i - j)
        return mask

    def forward(self, x):
        q = self.queries(x)
        k = self.keys(x)
        v = self.values(x)

        q_cos = q * self.mask_cos
        q_sin = q * self.mask_sin

        k_cos = k * self.mask_cos
        k_sin = k * self.mask_sin

        x = th.matmul(q_cos, k_cos.transpose(1, 2)) + th.matmul(q_sin, k_sin.transpose(1, 2))
        x = x * self.mask_decay
        x = th.matmul(x, v)

        return x


class SwishGate(nn.Module):
    def __init__(self, num_feature):
        super().__init__()
        self.ff1 = nn.Linear(num_feature, num_feature)
        self.ff2 = nn.Linear(num_feature, num_feature)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.ff1(x)
        x = x * self.sigmoid(x)
        x = x * y
        x = self.ff2(x)

        return x


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


class RetNetBlock(nn.Module):
    def __init__(self, num_sequence, num_feature, gamma):
        super().__init__()
        self.retention = ParallelRetention(num_sequence, num_feature, gamma)
        self.gate = SwishGate(num_feature)
        self.ffn = FeedForward(num_feature)
        self.norm = nn.LayerNorm(num_feature)

    def forward(self, x):
        y = self.norm(self.retention(x))
        y = self.gate(x, y) + x
        x = self.ffn(self.norm(y)) + y

        return x


class RetNet(nn.Module):
    def __init__(self, num_sequence, num_feature, gamma):
        super().__init__()
        self.retblock = RetNetBlock(num_sequence, num_feature, gamma)
        self.pool = nn.MaxPool2d([4, 4], 4)
        self.fc = nn.Linear(49, 10)

    def forward(self, x):
        x = self.retblock(x)
        x = self.pool(x)
        x = th.flatten(x, 1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    inputs = th.rand([10, 28, 28])
    model = RetNet(28, 28, 0.96875)
    print(model(inputs).shape)
