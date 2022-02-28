import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.quantized as quant

from collections import OrderedDict


def quantize(x, n_bits):
    a, b = x.max(), x.min()
    n_nums = (1 << n_bits)-1
    s = (a-b)/n_nums
    z = ((-b) / s).round()
    q = (x / s).round() + z
    return q, s, z


def dequantize(q, s, z):
    return s * (q - z)


class QuantizedLinear(nn.Module):
    """
    forward only
    when call state_dict, set keep_vals to True
    """

    def __init__(self, in_features, out_features, n_bits=8):
        super(QuantizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.ones(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = nn.Parameter(torch.ones(1))
        self.zero_point = nn.Parameter(torch.zeros(1))
        self.n_bits = n_bits

    def _forward(self, x):
        q_x, s_x, z_x = quantize(x, self.n_bits)
        q_w = self.weight
        s_w = self.scale
        z_w = self.zero_point
        b = self.bias

        x = F.linear(q_x-z_x, q_w-z_w, None)
        return (s_x*s_w)*x+b
        # wX = torch.matmul(q_x.to(torch.int)-z_x, (q_w.to(torch.int)-z_w))
        # return (s_x*s_w)*wX + b


class TestModule(nn.Module):
    def __init__(self, quantize=False):
        super(TestModule, self).__init__()
        self.quantize = quantize
        if quantize:
            self.linear = QuantizedLinear(6, 5)
        else:
            self.linear = nn.Linear(6, 5)

    def forward(self, x):
        x = self.linear._forward(x) if self.quantize else self.linear(x)
        return x


if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.randn(4, 6).to(device)
    m = TestModule().to(device)
    qm = TestModule(quantize=True).to(device)

    print(m(x))
    print(qm(x))
